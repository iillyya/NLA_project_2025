import math
import torch
from msign import msign


@torch.no_grad()
def manifold_muon(
    W: torch.Tensor,
    G: torch.Tensor,
    eta: float = 0.1,
    alpha: float = 0.01,
    dual_steps: int = 100,
    tol: float = 1e-6,
    msign_fn=msign,
    msign_steps: int | None = None,
) -> torch.Tensor:
    # Ensure that W and G are both tall matrices
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T

    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)

    # Ascend on the dual problem to find the update direction A
    for step in range(dual_steps):
        # Update the candidate direction A
        A = msign_fn(G + 2 * W @ Lambda, steps=msign_steps)

        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W

        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break

        # Update the dual variable
        Lambda -= alpha * (1.0 - step / dual_steps) * H

    # Descend on the primal problem
    new_W = W - eta * A

    # Retract to the manifold
    new_W = msign_fn(new_W, steps=msign_steps)
    # Restore the shape of the solution and return
    return new_W.T if should_transpose else new_W
