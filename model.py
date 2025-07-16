import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class SimpleSiameseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embedding_dim)
        )

    def forward_once(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


def train_model(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    epochs=10,
    scheduler=None,
    print_every=1
):
    """
    Generic PyTorch training loop.
    Args:
        model: torch.nn.Module
        train_loader: DataLoader yielding (input1, input2, label)
        optimizer: torch.optim.Optimizer
        loss_fn: loss function (e.g., ContrastiveLoss)
        device: torch.device
        epochs: number of epochs
        scheduler: optional learning rate scheduler
        print_every: print loss every N epochs
    """
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # Unpack batch
            if len(batch) == 3:
                x1, x2, labels = batch
            else:
                raise ValueError("train_loader must yield (x1, x2, labels) tuples")
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            # Forward
            out1, out2 = model(x1, x2)
            loss = loss_fn(out1, out2, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        if epoch % print_every == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    print("Training complete.")

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, emb1, emb2, label):
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss = (1 - label) * torch.pow(dist, 2) + \
               label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()