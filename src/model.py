import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


_MODEL_CONFIGS = {
    "small":  dict(n_embd=768,  n_layer=12, n_head=12, n_inner=3072),   # 124M
    "medium": dict(n_embd=1024, n_layer=24, n_head=16, n_inner=4096),   # 345M
}


class TSTModel(nn.Module):
    """GPT-2 with optional Token Superposition forward pass."""

    def __init__(self, size: str = "small"):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        cfg = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            **_MODEL_CONFIGS[size],
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        self.transformer = GPT2Model(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

    def forward(self, input_ids: torch.Tensor, bag_size: int = 1) -> torch.Tensor:
        use_ckpt = self.transformer.is_gradient_checkpointing
        if bag_size > 1:
            B, SL = input_ids.shape
            L = SL // bag_size
            raw = self.transformer.wte(input_ids.reshape(B * SL)).float()
            embeds = raw.reshape(B, L, bag_size, -1).mean(dim=2).to(raw.dtype)
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
            h = self.transformer.drop(embeds + self.transformer.wpe(pos))
            for block in self.transformer.h:
                if use_ckpt:
                    h = grad_checkpoint(block, h, use_reentrant=False)[0]
                else:
                    h = block(h)[0]
            return self.lm_head(self.transformer.ln_f(h))
        out = self.transformer(input_ids, use_cache=False)
        return self.lm_head(out.last_hidden_state)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bag_size: int = 1,
) -> torch.Tensor:
    """
    Multi-hot CE loss (MCE). bag_size=1 is standard next-token CE.
    logits:  (B, L, V)  — L = latent positions = data_seq / bag_size
    targets: (B, data_seq)
    Causality: s-token j predicts next bag of tokens at positions [j*s+s .. j*s+2s-1].
    """
    B, L, V = logits.shape
    if bag_size == 1:
        return F.cross_entropy(logits.reshape(B * L, V), targets.reshape(B * L))

    offset = bag_size - 1
    bags = (
        F.pad(targets, (0, offset), value=-100)[:, offset:]
        .reshape(B * L, bag_size)
    )
    # Compute log-sum-exp once over the vocabulary — one softmax per position, not bag_size.
    logits_flat = logits.reshape(B * L, V).float()
    log_z = torch.logsumexp(logits_flat, dim=-1)  # (B*L,)
    nll = torch.zeros(B * L, dtype=torch.float32, device=logits.device)
    for i in range(bag_size):
        idx = bags[:, i]          # (B*L,)
        valid = idx != -100
        nll[valid] += log_z[valid] - logits_flat[valid, idx[valid]]
    return (nll / bag_size).mean()
