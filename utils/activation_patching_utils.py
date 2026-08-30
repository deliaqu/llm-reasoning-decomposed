import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import os
from config import INSTRUCTION
from utils.shared_utils import untuple, make_hook, remove_all_hooks

def tokenize(question, model, tokenizer, answer_prefix="", print_result=False, instruction=None):
    instruction = instruction if instruction is not None else INSTRUCTION
    # Match evaluation format: question\ninstruction, apply_chat_template adds \n\n
    # after the assistant header — consistent with how the model was evaluated.
    messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = base + answer_prefix
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=12,  # Enough for "y" or "* y"
        do_sample=False,
        temperature=0,
    )
    # Extract only the model's continuation (after "x")
    input_length = inputs.input_ids.shape[1]
    answer = tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
    if print_result:
        print(prompt)
        print(f"Answer: {answer}")  # Output: "Total pencils: xy"
    
    return inputs, answer


def register_saving_hooks(model):
    """Register hooks at each layer of the model to get hidden states.

    Returns:
        state_dict: dictionary that stores all intermediate hidden states that can be used for patching
        state_hooks: dictionary that stores all the hook handles
    """
    
    state_dict, state_hooks = {}, []
    
    if model.config.model_type == "gpt_neox":
        hook_handle = model.gpt_neox.embed_in.register_forward_hook(
            make_hook(state_dict, f"embed_in")
        )
        state_hooks.append(hook_handle)

        for i in range(len(model.gpt_neox.layers)):
            hook_handle = model.gpt_neox.layers[i].register_forward_hook(
                make_hook(state_dict, f"layer_{i}")
            )
            state_hooks.append(hook_handle)

            hook_handle = model.gpt_neox.layers[i].attention.register_forward_hook(
                make_hook(state_dict, f"layer_attn_{i}")
            )
            state_hooks.append(hook_handle)

            hook_handle = model.gpt_neox.layers[i].mlp.register_forward_hook(
                make_hook(state_dict, f"layer_mlp_{i}")
            )
            state_hooks.append(hook_handle)

        hook_handle = model.gpt_neox.final_layer_norm.register_forward_hook(
            make_hook(state_dict, f"final_layer_norm")
        )
        state_hooks.append(hook_handle)

        hook_handle = model.gpt_neox.register_forward_hook(
            make_hook(state_dict, f"model")
        )
        state_hooks.append(hook_handle)

        hook_handle = model.embed_out.register_forward_hook(
            make_hook(state_dict, f"embed_out")
        )
        state_hooks.append(hook_handle)
    
    elif model.config.model_type in ['llama', 'mistral', 'qwen2']:
        hook_handle = model.model.embed_tokens.register_forward_hook(
            make_hook(state_dict, f"embed_in")
        )
        
        for i in range(len(model.model.layers)):
            hook_handle = model.model.layers[i].register_forward_hook(
                make_hook(state_dict, f"layer_{i}")
            )
            state_hooks.append(hook_handle)
            # fix here: for attn output patching
            hook_handle = model.model.layers[i].self_attn.register_forward_hook(
                make_hook(state_dict, f"layer_attn_output_{i}")
            )
            state_hooks.append(hook_handle)

            # fix here: for attn head patching
            hook_handle = model.model.layers[i].self_attn.o_proj.register_forward_hook(
                make_hook(state_dict, f"layer_attn_head_{i}")
            )
            state_hooks.append(hook_handle)

            hook_handle = model.model.layers[i].mlp.register_forward_hook(
                make_hook(state_dict, f"layer_mlp_{i}")
            )
            state_hooks.append(hook_handle)

        hook_handle = model.model.norm.register_forward_hook(
            make_hook(state_dict, f"final_layer_norm")
        )
        state_hooks.append(hook_handle)

        hook_handle = model.model.register_forward_hook(
            make_hook(state_dict, f"model")
        )
        state_hooks.append(hook_handle)

        hook_handle = model.lm_head.register_forward_hook(
            make_hook(state_dict, f"embed_out")
        )
        state_hooks.append(hook_handle)
        
        #state_hooks.append(hook_handle)
        
        
    else:
        print(f"Unknown model")

    return state_dict, state_hooks

def make_attn_hook(hs, indices=None, head_idx=None, n_heads=None):
    """Use with register_forward_pre_hook to patch inputs before projection."""
    def pre_hook(module, input):
        # input is a tuple; get the tensor
        input_tensor = input[0]  # shape: [batch, seq_len, n_heads * head_dim]
        batch_size, seq_len, hidden_size = input_tensor.shape
        
        # Reshape into heads
        head_dim = hidden_size // n_heads
        input_heads = input_tensor.view(batch_size, seq_len, n_heads, head_dim)
        hs_heads = untuple(hs).view(batch_size, -1, n_heads, head_dim)  # -1 for flexible seq_len
        
        # Patch specific head/positions
        if indices:
            for i in indices:
                input_heads[:, i, head_idx, :] = hs_heads[:, i, head_idx, :]
        
        # Reshape back and return
        
        input = (input_heads.view(batch_size, seq_len, hidden_size),)
        return input
        #patched_input = input_heads.view(batch_size, seq_len, hidden_size)
        #return (patched_input,)  # Must return a tuple
    
    return pre_hook


def make_patching_hook(hs, indices=None, head_idx=None, n_heads=None):
    """Patch target model by registering patching hooks.

    Args:
        hs: hidden states from source model used to patch target model.
        indices: list of token positions to patch (along sequence length).
        head_idx: if not None, patch only a specific attention head output.
    """

    def patching_hook(module, input, output):
        try:
            original_output = output
            output = untuple(output)
            hs_ = untuple(hs)

            if output.size(0) == output.size(1) == 1:
                return original_output

            indices_ = [indices] if isinstance(indices, int) else indices

            if head_idx is None:
                if indices_ is not None:
                    for i in indices_:
                        output[:, i] = hs_[:, i]
                else:
                    assert output.shape == hs_.shape, \
                        f"input: {untuple(input).shape}; output: {output.shape}; hs: {hs_.shape}"
                    output[:] = hs_

                return (output,) + original_output[1:] if isinstance(original_output, tuple) else output
            else:
                device = module.weight.device
                hs_ = untuple(hs).to(device)
                input_ = untuple(input).to(device)
                batch_size, seq_len, hidden_size = input_.shape

                head_dim = hidden_size // n_heads

                input_heads = input_.view(batch_size, seq_len, n_heads, head_dim)
                hs_heads = hs_.view(batch_size, -1, n_heads, head_dim)

                if indices_ is not None:
                    for i in indices_:
                        input_idx = i if i >= 0 else seq_len + i
                        source_idx = i if i >= 0 else hs_heads.shape[1] + i
                        if not (0 <= input_idx < seq_len and 0 <= source_idx < hs_heads.shape[1]):
                            raise IndexError(
                                f"Patch index {i} out of bounds for input seq_len={seq_len} "
                                f"and source seq_len={hs_heads.shape[1]}"
                            )
                        input_heads[:, i, head_idx, :] = hs_heads[:, i, head_idx, :]
                else:
                    raise ValueError("Token index must be provided when patching attention heads.")

                merged_input = input_heads.view(batch_size, -1, hidden_size)

                # Apply output projection: assume module is a linear layer (like o_proj)
                output = merged_input @ module.weight.t()
                if module.bias is not None:
                    output = output + module.bias

                return (output,) + original_output[1:] if isinstance(original_output, tuple) else output

        except Exception as e:
            # Special handling for dict-like output, e.g., LLaMA-style
            if isinstance(output, dict) and 'last_hidden_state' in output:
                output_ = output['last_hidden_state']
                hs_ = untuple(hs)

                if output_.size(0) == output_.size(1) == 1:
                    return output

                indices_ = [indices] if isinstance(indices, int) else indices

                if head_idx is None:
                    if indices_ is not None:
                        for i in indices_:
                            output_[:, i] = hs_[:, i]
                    else:
                        assert output_.shape == hs_.shape
                        output_[:] = hs_
                else:
                    device = module.weight.device
                    hs_ = untuple(hs).to(device)
                    input_ = untuple(input).to(device)
                    batch_size, seq_len, hidden_size = input_.shape
                    head_dim = hidden_size // n_heads

                    input_heads = input_.view(batch_size, seq_len, n_heads, head_dim)
                    hs_heads = hs_.view(batch_size, -1, n_heads, head_dim)

                    if indices_ is not None:
                        for i in indices_:
                            input_idx = i if i >= 0 else seq_len + i
                            source_idx = i if i >= 0 else hs_heads.shape[1] + i
                            if not (0 <= input_idx < seq_len and 0 <= source_idx < hs_heads.shape[1]):
                                raise IndexError(
                                    f"Patch index {i} out of bounds for input seq_len={seq_len} "
                                    f"and source seq_len={hs_heads.shape[1]}"
                                )
                            input_heads[:, i, head_idx, :] = hs_heads[:, i, head_idx, :]
                    else:
                        raise ValueError("Token index must be provided when patching attention heads.")

                    merged_input = input_heads.view(batch_size, -1, hidden_size)
                    device = module.weight.device
                    merged_input = merged_input.to(device)
                    output['last_hidden_state'] = merged_input @ module.weight.t()
                    if module.bias is not None:
                        output['last_hidden_state'] = output['last_hidden_state'] + module.bias

                return output
            else:
                raise RuntimeError(f"Patching hook failed: {e}")

    return patching_hook

def calculate_logprob_changes(
    clean_inputs,
    corrupted_inputs,
    tokenizer,
    clean_model,
    corrupted_model,
    target_tokens, 
    num_layers=32 + 2,
    patching_scope="layer", # layer, mlp, attn_head, attn_output
    topk=10,  # <--- NEW: topk parameter
):
    remove_all_hooks(clean_model)
    remove_all_hooks(corrupted_model)
    
    clean_inputs = clean_inputs.to(clean_model.device)
    corrupted_inputs = corrupted_inputs.to(corrupted_model.device)
    
    clean_input_ids = clean_inputs["input_ids"]
    clean_attention_mask = clean_inputs["attention_mask"]

    corrupted_input_ids = corrupted_inputs["input_ids"]
    corrupted_attention_mask = corrupted_inputs["attention_mask"]
    
    clean_model.eval()
    corrupted_model.eval()
    
    # === CLEAN MODEL ===
    state_dict, state_hooks = register_saving_hooks(clean_model)
    
    clean_input_ids = clean_input_ids.clone()
    
    with torch.no_grad():
        clean_outputs = clean_model(input_ids=clean_input_ids, attention_mask=clean_attention_mask)
        clean_logits = clean_outputs.logits[:, -1, :]
        clean_probs = F.log_softmax(clean_logits, dim=-1)
        
        clean_token = torch.argmax(clean_probs, dim=-1)
        clean_logprobs = clean_probs[0, clean_token.item()].item()
    
    target_tokens = target_tokens if target_tokens else [[clean_token]]
    clean_target_logprobs = np.array([clean_probs[0, tokens].max().item() for tokens in target_tokens])
    
    # === CORRUPTED MODEL ===
    remove_all_hooks(corrupted_model)
    remove_all_hooks(clean_model)
    
    corrupted_input_ids = corrupted_input_ids.clone()
    
    with torch.no_grad():
        corrupted_outputs = corrupted_model(input_ids=corrupted_input_ids, attention_mask=corrupted_attention_mask)
        corrupted_logits = corrupted_outputs.logits[:, -1, :]
        corrupted_probs = F.log_softmax(corrupted_logits, dim=-1)
        corrupted_token = torch.argmax(corrupted_probs, dim=-1)
        corrupted_logprobs = corrupted_probs[0, corrupted_token.item()].item()
    
    corrupted_target_logprobs = np.array([corrupted_probs[0, tokens].max().item() for tokens in target_tokens])
    
    # === RESTORATION MODEL ===
    seq_length = 1
    num_heads = corrupted_model.config.num_attention_heads if patching_scope == "attn_head" else 1
    
    patched_target_logprobs = np.full((num_layers, num_heads, seq_length, len(target_tokens)), None)
    patched_predictions = np.full((num_layers, num_heads, seq_length), None)
    patched_topk_predictions = np.full((num_layers, num_heads, seq_length, topk), None)  # <--- NEW
    
    token_idx = -1
    
    for layer_idx in range(num_layers):
        num_heads_in_this_layer = corrupted_model.config.num_attention_heads if patching_scope == "attn_head" else 1
        
        for head_idx in range(num_heads_in_this_layer):
            remove_all_hooks(corrupted_model)
        
            if corrupted_model.config.model_type == "gpt_neox":
                if 0 <= layer_idx < num_layers - 2:
                    if patching_scope == "layer":
                        hook_handle = corrupted_model.gpt_neox.layers[layer_idx].register_forward_hook(
                            make_patching_hook(state_dict[f'layer_{layer_idx}'], [token_idx])
                        )
                    elif patching_scope == "mlp":
                        hook_handle = corrupted_model.gpt_neox.layers[layer_idx].mlp.register_forward_hook(
                            make_patching_hook(state_dict[f'layer_mlp_{layer_idx}'], [token_idx])
                        )
                    elif patching_scope == "attn":
                        hook_handle = corrupted_model.gpt_neox.layers[layer_idx].attention.register_forward_hook(
                            make_patching_hook(state_dict[f"layer_attn_{layer_idx}"], [token_idx], [head_idx])
                        )
                    else:
                        raise Exception("Unknown Patching Scope!")
                elif layer_idx == num_layers - 2:
                    hook_handle = corrupted_model.gpt_neox.final_layer_norm.register_forward_hook(
                        make_patching_hook(state_dict['final_layer_norm'], [token_idx])
                    )
                elif layer_idx == num_layers - 1:
                    hook_handle = corrupted_model.embed_out.register_forward_hook(
                        make_patching_hook(state_dict['embed_out'], [token_idx])
                    )

            elif corrupted_model.config.model_type in ["llama", "mistral", "qwen2"]:
                if 0 <= layer_idx < num_layers - 2:
                    if patching_scope == "layer":
                        hook_handle = corrupted_model.model.layers[layer_idx].register_forward_hook(
                            make_patching_hook(state_dict[f'layer_{layer_idx}'], [token_idx])
                        )
                    elif patching_scope == "mlp":
                        hook_handle = corrupted_model.model.layers[layer_idx].mlp.register_forward_hook(
                            make_patching_hook(state_dict[f'layer_mlp_{layer_idx}'], [token_idx])
                        )
                    elif patching_scope == "attn_head": # patching attn head
                        hook_handle = corrupted_model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                            make_patching_hook(state_dict[f"layer_attn_head_{layer_idx}"], [token_idx], [head_idx],corrupted_model.config.num_attention_heads)
                        )
                    elif patching_scope == "attn_output": # patching attn layer output
                        hook_handle = corrupted_model.model.layers[layer_idx].self_attn.register_forward_hook(
                            make_patching_hook(state_dict[f"layer_attn_output_{layer_idx}"], [token_idx])
                        )
                    else:
                        raise Exception("Unknown Patching Scope!")
                elif layer_idx == num_layers - 2:
                    hook_handle = corrupted_model.model.norm.register_forward_hook(
                        make_patching_hook(state_dict['final_layer_norm'], [token_idx])
                    )
                elif layer_idx == num_layers - 1:
                    hook_handle = corrupted_model.lm_head.register_forward_hook(
                        make_patching_hook(state_dict['embed_out'], [token_idx])
                    )
            else:
                raise Exception("Unknown Model Type!")

            with torch.no_grad():
                patched_outputs = corrupted_model(input_ids=corrupted_input_ids, attention_mask=corrupted_attention_mask)
                patched_logits = patched_outputs.logits[:, -1, :]
                patched_probs = F.log_softmax(patched_logits, dim=-1)
                patched_token = torch.argmax(patched_probs, dim=-1)

                topk_probs, topk_indices = torch.topk(patched_probs, k=topk, dim=-1)  # <--- NEW
                
                if torch.isnan(patched_logits).any() or torch.isinf(patched_logits).any():
                    print("Pathching Logits encountered nan or inf at layer ", layer_idx )
                
                if torch.isnan(patched_probs).any() or torch.isinf(patched_logits).any():
                    print("Pathching Logprobs encountered nan or inf at layer ", layer_idx )
    

            for i, tokens in enumerate(target_tokens):
                patched_target_logprobs[layer_idx, head_idx, 0, i] = patched_probs[0, tokens].max().item()

            patched_predictions[layer_idx, head_idx, 0] = patched_token.item()
            patched_topk_predictions[layer_idx, head_idx, 0] = topk_indices[0].cpu().numpy()  # <--- NEW
            
    remove_all_hooks(clean_model)
    remove_all_hooks(corrupted_model)
    
    for handle in state_hooks:
        handle.remove()
    
    torch.cuda.empty_cache()
    
    return clean_token, clean_logprobs, clean_target_logprobs, \
        corrupted_token, corrupted_logprobs, corrupted_target_logprobs, \
        patched_target_logprobs, patched_predictions, patched_topk_predictions  # <--- UPDATED
