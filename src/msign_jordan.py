import torch

JORDAN_ABC: tuple[float, float, float] = (
    3.4445,   # a  (пример формата; НЕ гарантированно точные числа)
    -4.7750,  # b
    2.0315,   # c
)

@torch.no_grad()
def msign_jordan5(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    assert G.ndim >= 2
    should_transpose = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    a, b, c = JORDAN_ABC

    for _ in range(steps):
        s = x @ x.mT
        s2 = s @ s
        x = (a * x) + (b * (s @ x)) + (c * (s2 @ x))

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()
