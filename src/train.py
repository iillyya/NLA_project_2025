import argparse
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from hyperspherical_descent import hyperspherical_descent
from manifold_muon import manifold_muon
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ===================== msign variants (nothing else depends on them) =====================

ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


@torch.no_grad()
def msign_polarexpress(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Polar Express (degree=5)"""
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x /= (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()


@torch.no_grad()
def msign_ns3(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Newton–Schulz degree-3: X <- 1.5X - 0.5 X X^T X"""
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x /= (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for _ in range(steps):
        s = x @ x.mT
        x = (1.5 * x) - (0.5 * (s @ x))

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()


@torch.no_grad()
def msign_ns5(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Newton–Schulz degree-5: X <- (15/8)X + (-10/8)S X + (3/8)S^2 X"""
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x /= (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    a, b, c = (15.0 / 8.0), (-10.0 / 8.0), (3.0 / 8.0)

    for _ in range(steps):
        s = x @ x.mT
        s2 = s @ s
        x = (a * x) + (b * (s @ x)) + (c * (s2 @ x))

    if should_transpose:
        x = x.mT
    return torch.nan_to_num(x).float()


# ===================== data/model (unchanged) =====================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                         (0.24703233, 0.24348505, 0.26158768))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ===================== training/eval (minimal change: update_kwargs) =====================

def train(epochs, initial_lr, update, wd, update_kwargs=None):
    model = MLP().cuda()
    criterion = nn.CrossEntropyLoss()

    update_kwargs = update_kwargs or {}

    if update == AdamW:
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=wd)
    else:
        assert update in [manifold_muon, hyperspherical_descent]
        optimizer = None

    steps = epochs * len(train_loader)
    step = 0

    if optimizer is None:
        # Project the weights to the manifold
        for p in model.parameters():
            p.data = update(p.data, torch.zeros_like(p.data), eta=0, **update_kwargs)

    epoch_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            lr = initial_lr * (1 - step / steps)
            with torch.no_grad():
                if optimizer is None:
                    for p in model.parameters():
                        p.data = update(p, p.grad, eta=lr, **update_kwargs)
                else:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    optimizer.step()
            step += 1

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        end_time = time.time()
        epoch_loss = running_loss / len(train_loader)
        epoch_time = end_time - start_time
        epoch_losses.append(epoch_loss)
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Time: {epoch_time:.4f} seconds")
    return model, epoch_losses, epoch_times


def eval(model):
    model.eval()
    with torch.no_grad():
        accs = []
        for dataloader in [test_loader, train_loader]:
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accs.append(100 * correct / total)

    print(f"Accuracy of the network on the {len(test_loader.dataset)} test images: {accs[0]} %")
    print(f"Accuracy of the network on the {len(train_loader.dataset)} train images: {accs[1]} %")
    return accs


def weight_stats(model):
    singular_values = []
    norms = []
    for p in model.parameters():
        u, s, v = torch.svd(p)
        singular_values.append(s)
        norms.append(p.norm())
    return singular_values, norms


# ===================== main (minimal change: CLI for msign + update_kwargs) =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--update", type=str, default="manifold_muon",
                        choices=["manifold_muon", "hyperspherical_descent", "adam"],
                        help="Update rule to use.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for AdamW.")

    # Only used when --update manifold_muon
    parser.add_argument("--msign", type=str, default="polarexpress",
                        choices=["polarexpress", "ns3", "ns5"],
                        help="Which msign() implementation to use (only for manifold_muon).")
    parser.add_argument("--msign_steps", type=int, default=5,
                        help="Number of iterations inside msign (only for manifold_muon).")

    args = parser.parse_args()

    # determinism flags
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    update_rules = {
        "manifold_muon": manifold_muon,
        "hyperspherical_descent": hyperspherical_descent,
        "adam": AdamW
    }
    update = update_rules[args.update]

    # Choose msign variant (only for manifold_muon)
    msign_map = {
        "polarexpress": msign_polarexpress,
        "ns3": msign_ns3,
        "ns5": msign_ns5,
    }
    update_kwargs = {}
    if args.update == "manifold_muon":
        update_kwargs = {
            "msign_fn": msign_map[args.msign],
            "msign_steps": args.msign_steps,
        }

    print(f"Training with: {args.update}")
    if args.update == "manifold_muon":
        print(f"msign: {args.msign}  msign_steps: {args.msign_steps}")
    print(f"Epochs: {args.epochs} --- LR: {args.lr}", f"--- WD: {args.wd}" if args.update == "adam" else "")

    model, epoch_losses, epoch_times = train(
        epochs=args.epochs,
        initial_lr=args.lr,
        update=update,
        wd=args.wd,
        update_kwargs=update_kwargs,
    )
    test_acc, train_acc = eval(model)
    singular_values, norms = weight_stats(model)

    results = {
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "wd": args.wd,
        "update": args.update,
        "msign": args.msign if args.update == "manifold_muon" else None,
        "msign_steps": args.msign_steps if args.update == "manifold_muon" else None,
        "epoch_losses": epoch_losses,
        "epoch_times": epoch_times,
        "test_acc": test_acc,
        "train_acc": train_acc,
        "singular_values": singular_values,
        "norms": norms
    }

    filename = (
        f"update-{args.update}"
        + (f"-msign-{args.msign}-msignsteps-{args.msign_steps}" if args.update == "manifold_muon" else "")
        + f"-lr-{args.lr}-wd-{args.wd}-seed-{args.seed}.pkl"
    )
    os.makedirs("results", exist_ok=True)
    outpath = os.path.join("results", filename)

    print(f"Saving results to {outpath}")
    with open(outpath, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {outpath}")
