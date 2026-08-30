import torch
from utils.shared_utils import untuple

def make_hook(states_dict, layer_name):
    def hook(module, input, output):
        states_dict[layer_name] = {
            "input": untuple(input).detach().cpu(),
            "output": untuple(output).detach().cpu()
        }
    return hook

def register_saving_hooks(model):
    """
    Register hooks on model to capture:
    - input, output to decoder layer
    - Attention output (via o_proj)
    - MLP output (via down_proj)
    Returns:
        state_dict: dict storing hook outputs
        hook_handles: list of hook handles (removable)
    """
    state_dict = {}
    hook_handles = []

    for i, block in enumerate(model.model.layers):
        # Capture input to the full layer (via pre_hook on block)
        
        hook_handles.append(
            block.register_forward_hook(
                make_hook(state_dict, f"layer_{i}")
            )
        )

        # o_proj gives the output of attention (pre-residual addition)
        hook_handles.append(
            block.self_attn.o_proj.register_forward_hook(
                make_hook(state_dict, f"attn_out_{i}")
            )
        )

        # Capture MLP output (pre-residual addition)
        # MLP(x)=down_proj(SiLU(gate_proj(x))⊙ up_proj(x)) where
        # gate and up are two parallel linear projections
        # ⊙ denotes element-wise multiplication.
    
        hook_handles.append(
            block.mlp.down_proj.register_forward_hook(
                make_hook(state_dict, f"mlp_out_{i}")
            )
        )
        # the difference between this and the intermediate resid is that
        # mlp input is a normalzied version of the mid residual stream
        # preprocessing and prepare the inputs for MLP
        # in some sense, this is also important? 
        # to get "useful" information for the MLP among all info available in the resid stream
        
        hook_handles.append(
            block.mlp.register_forward_hook(
                make_hook(state_dict, f"mlp_in_{i}")
            )
        )

    return state_dict, hook_handles

def remove_all_hooks(handles):
    for h in handles:
        h.remove()
        
def generate_with_hidden_logging(
    model,
    tokenizer,
    input_ids,
    terminators,
    max_new_tokens=512,
):
    collected_logs = {
        "attn_out": [],      # (T, L, H)
        "mlp_in": [],       # (T, L, H)
        "mlp_out": [],       # (T, L, H)
        "resid_mid": [],     # (T, L, H)
        "resid_final": []    # (T, L, H)
    }

    model.eval()

    generated_ids = input_ids["input_ids"]
    attention_mask = input_ids["attention_mask"]

    for _ in range(max_new_tokens):
        state_dict, hook_handles = register_saving_hooks(model)

        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            logits = model.lm_head(hidden_states[-1][:, -1, :])
            next_token = torch.argmax(logits, dim=-1)

        # Residual stream (final)
        resid_final = torch.stack([
            hidden_states[i + 1][:, -1, :].squeeze(0).detach().cpu()
            for i in range(len(model.model.layers))
        ])  # (L, H)

        resid_mid = []
        attn_out = []
        mlp_out = []
        mlp_in = []

        for i in range(len(model.model.layers)):
            x = state_dict[f"layer_{i}"]["input"][:, -1, :].squeeze(0)
            attn = state_dict[f"attn_out_{i}"]["output"][:, -1, :].squeeze(0)
            mlp = state_dict[f"mlp_out_{i}"]["output"][:, -1, :].squeeze(0)
            mlp_inp = state_dict[f"mlp_in_{i}"]["input"][:, -1, :].squeeze(0)
            
            resid_mid.append((x + attn).cpu())
            attn_out.append(attn.cpu())
            mlp_out.append(mlp.cpu())
            mlp_in.append(mlp_inp.cpu())

        collected_logs["resid_final"].append(resid_final)       # (L, H)
        collected_logs["resid_mid"].append(torch.stack(resid_mid))  # (L, H)
        collected_logs["attn_out"].append(torch.stack(attn_out))    # (L, H)
        collected_logs["mlp_out"].append(torch.stack(mlp_out))      # (L, H)
        collected_logs["mlp_in"].append(torch.stack(mlp_in))      # (L, H)

        remove_all_hooks(hook_handles)

        # Update inputs
        next_token_id = next_token[:, None].to(generated_ids.device)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)

        if next_token.item() in terminators:
            break

    stacked_result = {k: torch.stack(v) for k, v in collected_logs.items()}  # (T, L, H)
    return generated_ids, stacked_result