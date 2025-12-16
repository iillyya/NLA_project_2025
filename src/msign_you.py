import torch

YOU_ABC_LIST: list[tuple[float, float, float]] = [
    # заполни 6 троек (a,b,c)
]

@torch.no_grad()
def msign_you6(G: torch.Tensor, steps: int = 6) -> torch.Tensor:
    assert G.ndim >= 2
    assert steps == 6, "You method is defined for 6 steps in this implementation."
    assert len(YOU_ABC_LIST) == 6, "Fill YOU_ABC_LIST with exactly 6 (a,b,c) tuples."

    should_transpose = G.size(-2) > G.size(-1)
    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for i in range(6):
        a, b, c = YOU_ABC_LIST[i]
        s = x @ x.mT
        s2 = s @ s
        x = (a * x) + (b * (s @ x)) + (c * (s2 @ x))

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()
