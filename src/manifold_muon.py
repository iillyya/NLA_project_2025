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

    # Helper: call msign with/without steps depending on what the function supports
    def _msign(X: torch.Tensor) -> torch.Tensor:
        if msign_steps is None:
            return msign_fn(X)
        try:
            return msign_fn(X, steps=msign_steps)
        except TypeError:
            # In case msign_fn does not accept `steps`
            return msign_fn(X)

    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)

    # Ascend on the dual problem to find the update direction A
    for step in range(dual_steps):
        # Update the candidate direction A
        A = _msign(G + 2 * W @ Lambda)

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
    new_W = _msign(new_W)

    # Restore the shape of the solution and return
    return new_W.T if should_transpose else new_W
