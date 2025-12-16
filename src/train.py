import argparse
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

# new msigns (you must have these python files / functions)
from msign import msign as msign_polarexpress
from msign_ns5 import msign_ns5
from msign_jordan import msign_jordan5
from msign_svd_polar import msign_svd_ref
# from msign_you import msign_you6   # uncomment when YOU_ABC_LIST is filled


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

            outputs = model(images)
            loss = criterion(outputs, labels)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--update", type=str, default="manifold_muon",
                        choices=["manifold_muon", "hyperspherical_descent", "adam"],
                        help="Update rule to use.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for AdamW.")

    # only for manifold_muon
    parser.add_argument("--msign", type=str, default="polarexpress",
                        choices=["polarexpress", "ns5", "jordan5", "svd_ref"],  # add "you6" when ready
                        help="Which msign() to use (only for manifold_muon).")
    parser.add_argument("--msign_steps", type=int, default=5,
                        help="Number of iterations inside msign (only for polynomial msigns).")

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

    msign_map = {
        "polarexpress": msign_polarexpress,
        "ns5": msign_ns5,
        "jordan5": msign_jordan5,
        "svd_ref": msign_svd_ref,
        # "you6": msign_you6,
    }

    update_kwargs = {}
    if args.update == "manifold_muon":
        # If your manifold_muon signature supports msign_steps, pass it.
        # If not, remove msign_steps here and bake steps into the function itself.
        update_kwargs = {
            "msign_fn": msign_map[args.msign],
            "msign_steps": args.msign_steps,
        }

    print(f"Training with: {args.update}")
    if args.update == "manifold_muon":
        print(f"msign: {args.msign} --- msign_steps: {args.msign_steps}")
    print(f"Epochs: {args.epochs} --- LR: {args.lr}", f"--- WD: {args.wd}" if args.update == "adam" else "")

    model, epoch_losses, epoch_times = train(
        epochs=args.epochs,
        initial_lr=args.lr,
        update=update,
        wd=args.wd,
        update_kwargs=update_kwargs
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
