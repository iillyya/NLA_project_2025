import torch

YOU_ABC_LIST = [
    (3955/1024, -8306/1024, 5008/1024),
    (3735/1024, -6681/1024, 3463/1024),
    (3799/1024, -6499/1024, 3211/1024),
    (4019/1024, -6385/1024, 2906/1024),
    (2677/1024, -3029/1024, 1162/1024),
    (2172/1024, -1833/1024,  682/1024),
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
