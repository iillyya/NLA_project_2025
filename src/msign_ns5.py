import torch

@torch.no_grad()
def msign_ns5(G: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """
    Degree-5 Newton–Schulz:
    X <- (15/8)X - (10/8) S X + (3/8) S^2 X , where S = X X^T
    """
    assert G.ndim >= 2
    should_transpose = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    a, b, c = (15.0/8.0), (-10.0/8.0), (3.0/8.0)

    for _ in range(steps):
        s = x @ x.mT          # S
        s2 = s @ s            # S^2
        x = (a * x) + (b * (s @ x)) + (c * (s2 @ x))

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()
