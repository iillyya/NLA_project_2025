
import torch
@torch.no_grad()
def msign_svd(G: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """
    Q = U V^T (точный полярный фактор) — годится как reference на маленьких матрицах.
    """
    assert G.ndim >= 2
    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return U @ Vh
