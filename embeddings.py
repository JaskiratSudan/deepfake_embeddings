import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_embeddings(model, data_list, device='cpu', batch_size=32):
    """
    data_list: list of torch tensors (e.g., mel spectrograms)
    Returns: numpy array of embeddings, list of indices
    """
    model.eval()
    model.to(device)
    loader = DataLoader(data_list, batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.forward_once(batch)
            if isinstance(emb, tuple):  # If model returns (embedding, features)
                emb = emb[0]
            embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings



def plot_tsne(embeddings, labels, title="t-SNE of Embeddings", plot_3d=False, save_path=None):
    """
    embeddings: numpy array [N, embedding_dim]
    labels: list of class labels (e.g., 'real', 'fake')
    plot_3d: if True, plot interactive 3D plotly plot; else, 2D matplotlib plot
    save_path: if provided and plot_3d=True, saves the interactive plot as HTML
    """
    if plot_3d:
        from sklearn.manifold import TSNE
        import plotly.express as px
        tsne = TSNE(n_components=3, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        fig = px.scatter_3d(
            x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
            color=labels,
            title=title,
            labels={'color': 'Class'}
        )
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Interactive plot saved: {save_path}")
        fig.show()
    else:
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        label_set = sorted(set(labels))
        color_map = {l: c for l, c in zip(label_set, ['blue', 'red', 'green', 'orange', 'purple'])}
        plt.figure(figsize=(8,6))
        for l in label_set:
            idx = [i for i, lab in enumerate(labels) if lab == l]
            plt.scatter(reduced[idx,0], reduced[idx,1], c=color_map[l], label=l, alpha=0.6)
        plt.legend()
        plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.show()