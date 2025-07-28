import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from embeddings import *

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
    # --- New parameters for plotting ---
    val_data_list=None,
    val_labels=None,
    plot_every=1,
    # ------------------------------------
    epochs=10,
    scheduler=None
):
    """
    Training loop that generates a 2D t-SNE plot of validation embeddings every N epochs.
    """
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in progress_bar:
            x1, x2, labels = batch
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            out1, out2 = model(x1, x2)
            loss = loss_fn(out1, out2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            current_avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{current_avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] Summary - Loss: {avg_loss:.4f}")

        # --- Block to generate and plot embeddings ---
        if val_data_list and val_labels and (epoch % plot_every == 0):
            print(f"Generating t-SNE plot for epoch {epoch}...")
            # Use the existing get_embeddings function
            embeddings = get_embeddings(model, val_data_list, device=device)
            
            # Use the existing plot_tsne function for a 2D plot
            plot_tsne(
                embeddings, 
                val_labels, 
                title=f"t-SNE of Validation Embeddings after Epoch {epoch}",
                plot_3d=False
            )
        # ---------------------------------------------

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