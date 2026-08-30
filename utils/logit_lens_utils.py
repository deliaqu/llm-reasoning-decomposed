import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from utils.shared_utils import untuple, remove_all_hooks
from utils.activation_patching_utils import register_saving_hooks


# ---------------------------------------------------------------------------
# Tuned lens translator
# ---------------------------------------------------------------------------

class TunedLensTranslator(nn.Module):
    """Low-rank affine translator: T(h) = h + U @ (V @ h) + b."""

    def __init__(self, d_model: int, rank: int = 64):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.empty(rank, d_model))
        nn.init.normal_(self.V, std=0.02)
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.V.device != h.device:
            self.to(h.device)
        return h + (h @ self.V.T) @ self.U.T + self.b


def load_translators(model_name: str, device="cpu") -> list[TunedLensTranslator] | None:
    """Load per-layer tuned lens translators from disk.

    Returns a list of translators (one per layer) or None if weights not found.
    """
    from config import CACHE_DIR
    translator_dir = Path(CACHE_DIR) / "tuned_lens" / "translators" / model_name

    meta_path = translator_dir / "meta.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    d_model, rank, n_layers = meta["d_model"], meta["rank"], meta["n_layers"]

    translators = []
    for i in range(n_layers):
        t = TunedLensTranslator(d_model, rank)
        t.load_state_dict(torch.load(translator_dir / f"layer_{i}.pt",
                                     weights_only=True))
        t.to(device).eval()
        translators.append(t)

    return translators


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------

def get_logit_lens(state_dict, model, target_token_ids,
                   translators: list[TunedLensTranslator] | None = None):
    """Apply norm + lm_head to each layer's last-token hidden state.

    If translators are provided, applies T_l(h_l) before norm + lm_head
    (tuned lens). Otherwise falls back to the standard logit lens.

    Returns raw logits of shape (num_layers, len(target_token_ids)).
    """
    num_layers = model.config.num_hidden_layers
    result = np.full((num_layers, len(target_token_ids)), float("nan"))

    with torch.no_grad():
        for i in range(num_layers):
            key = f"layer_{i}"
            if key not in state_dict:
                continue
            h = untuple(state_dict[key])[:, -1, :]  # (1, hidden_dim)
            if translators is not None:
                h = translators[i](h.float()).to(h.dtype)
            normed = model.model.norm(h)
            logits = model.lm_head(normed)           # (1, vocab_size)
            result[i] = [logits[0, tid].item() for tid in target_token_ids]

    return result  # (num_layers, n_targets)


def get_layer_kl(sym_state, noop_state, model,
                 translators: list[TunedLensTranslator] | None = None):
    """Compute KL(P_sym || P_noop) over the full vocabulary at each layer.

    Returns shape (num_layers,) in nats.
    """
    import torch.nn.functional as F
    num_layers = model.config.num_hidden_layers
    result = np.full(num_layers, float("nan"))

    with torch.no_grad():
        for i in range(num_layers):
            key = f"layer_{i}"
            if key not in sym_state or key not in noop_state:
                continue
            h_sym  = untuple(sym_state[key])[:, -1, :]
            h_noop = untuple(noop_state[key])[:, -1, :]
            if translators is not None:
                h_sym  = translators[i](h_sym.float()).to(h_sym.dtype)
                h_noop = translators[i](h_noop.float()).to(h_noop.dtype)
            log_p_sym  = F.log_softmax(model.lm_head(model.model.norm(h_sym)),  dim=-1)
            log_p_noop = F.log_softmax(model.lm_head(model.model.norm(h_noop)), dim=-1)
            # KL(P_sym || P_noop) = sum P_sym * (log P_sym - log P_noop)
            result[i] = F.kl_div(log_p_noop, log_p_sym,
                                  log_target=True, reduction="sum").item()

    return result  # (num_layers,)


# ---------------------------------------------------------------------------
# Component attribution
# ---------------------------------------------------------------------------

def get_component_attribution(state_dict, model, correct_id, wrong_id,
                               translators: list[TunedLensTranslator] | None = None):
    """Decompose the logit difference into per-layer component contributions.

    For attn and mlp: W_U_diff · component_output (direct attribution, no norm).
    For resid_mid: norm + lm_head on the intermediate residual (after attn,
    before MLP); optionally translated if tuned lens translators are provided.

    Returns dict:
      attn_attr  — (num_layers,)     W_U_diff · attn_output
      mlp_attr   — (num_layers,)     W_U_diff · mlp_output
      resid_mid  — (num_layers, 2)   logits[:, 0]=correct, [:, 1]=wrong at resid_mid
    """
    num_layers   = model.config.num_hidden_layers
    target_ids   = [correct_id, wrong_id]

    W_U      = model.lm_head.weight.float()                     # (vocab, hidden)
    W_U_diff = (W_U[correct_id] - W_U[wrong_id]).detach()       # (hidden,)

    attn_attr = np.full(num_layers, float("nan"))
    mlp_attr  = np.full(num_layers, float("nan"))
    resid_mid = np.full((num_layers, 2), float("nan"))

    with torch.no_grad():
        for i in range(num_layers):
            attn_key = f"layer_attn_output_{i}"
            mlp_key  = f"layer_mlp_{i}"

            if attn_key in state_dict:
                attn_out = untuple(state_dict[attn_key])[:, -1, :].float().squeeze(0).to(W_U_diff.device)
                attn_attr[i] = (W_U_diff @ attn_out).item()

            if mlp_key in state_dict:
                mlp_out = untuple(state_dict[mlp_key])[:, -1, :].float().squeeze(0).to(W_U_diff.device)
                mlp_attr[i] = (W_U_diff @ mlp_out).item()

            if attn_key in state_dict:
                prev_key = "embed_in" if i == 0 else f"layer_{i - 1}"
                if prev_key in state_dict:
                    norm_dev = model.model.norm.weight.device
                    resid_pre_h = untuple(state_dict[prev_key])[:, -1, :].to(norm_dev)
                    attn_out_h  = untuple(state_dict[attn_key])[:, -1, :].to(norm_dev)
                    h_mid       = resid_pre_h + attn_out_h
                    # No translator here: T_l was trained on the full layer output
                    # (resid_pre + attn + mlp), not this mid-layer point.
                    normed  = model.model.norm(h_mid)
                    logits  = model.lm_head(normed.to(model.lm_head.weight.device))
                    resid_mid[i] = [logits[0, tid].item() for tid in target_ids]

    return {
        "attn_attr": attn_attr,
        "mlp_attr":  mlp_attr,
        "resid_mid": resid_mid,  # (num_layers, 2): [:, 0]=correct, [:, 1]=wrong
    }


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------

def forward_with_hooks(model, inputs):
    """Register saving hooks, run a single forward pass, return state_dict."""
    remove_all_hooks(model)
    state_dict, state_hooks = register_saving_hooks(model)
    with torch.no_grad():
        model(**{k: v for k, v in inputs.items()})
    for h in state_hooks:
        h.remove()
    remove_all_hooks(model)
    return state_dict
