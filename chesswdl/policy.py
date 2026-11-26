"""정책 공통 규칙. (Drive chess/chesswdl 의 추론용 사본)"""
import torch
import torch.nn.functional as F

DEFAULT_TOP_P = 0.90


def mask_illegal(pol_logits, legal):
    if legal is None:
        return pol_logits.float()
    mask = torch.zeros(pol_logits.shape, dtype=torch.bool, device=pol_logits.device)
    mask.scatter_(-1, legal.clamp_min(0), legal >= 0)
    return pol_logits.float().masked_fill(~mask, -1e9)


def select_candidates(probs, top_p=DEFAULT_TOP_P):
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p는 (0, 1] 범위여야 합니다.")
    p_sorted, idx = probs.double().sort(descending=True)
    cum = p_sorted.cumsum(0)
    before = cum - p_sorted
    if top_p >= 1.0:
        n = int((p_sorted > 0).sum())
    else:
        n = int(((before < top_p) & (p_sorted > 0)).sum())
    n = max(n, 1)
    return idx[:n].tolist(), p_sorted[:n].tolist(), float(cum[n - 1])


def policy_nll(masked_logits, target):
    return F.cross_entropy(masked_logits, target, reduction="none")
