
import torch

YOU_ABC_LIST: list[tuple[float, float, float]] = [
    # (a1, b1, c1),
    # (a2, b2, c2),
    # ...
    # (a6, b6, c6),
]

@torch.no_grad()
def msign_you6(G: torch.Tensor, steps: int = 6) -> torch.Tensor:
    assert G.ndim >= 2
    assert len(YOU_ABC_LIST) > 0, "Fill YOU_ABC_LIST with 6 (a,b,c) tuples."
    should_transpose = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for i in range(steps):
        a, b, c = YOU_ABC_LIST[i] if i < len(YOU_ABC_LIST) else YOU_ABC_LIST[-1]
        s = x @ x.mT
        s2 = s @ s
        x = (a * x) + (b * (s @ x)) + (c * (s2 @ x))

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()
