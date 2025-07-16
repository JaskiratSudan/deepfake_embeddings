import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset

def make_positive_negative_pairs(real_list, fake_list, num_pairs):
    """
    Create positive (same class) and negative (different class) pairs.
    Returns:
        pairs: list of (item1, item2)
        labels: list of 0 (positive) or 1 (negative)
        classes: list of (class1, class2) for each pair
    """
    pairs = []
    labels = []
    classes = []

    num_positive = num_pairs // 2
    num_negative = num_pairs - num_positive

    # Positive pairs (same class)
    for _ in range(num_positive // 2):
        if len(real_list) >= 2:
            a, b = random.sample(real_list, 2)
            pairs.append((a, b))
            labels.append(0)
            classes.append(('real', 'real'))
        if len(fake_list) >= 2:
            a, b = random.sample(fake_list, 2)
            pairs.append((a, b))
            labels.append(0)
            classes.append(('fake', 'fake'))

    # Negative pairs (different class)
    for _ in range(num_negative):
        a = random.choice(real_list)
        b = random.choice(fake_list)
        if random.random() < 0.5:
            pairs.append((a, b))
            classes.append(('real', 'fake'))
        else:
            pairs.append((b, a))
            classes.append(('fake', 'real'))
        labels.append(1)

    # Shuffle all together
    combined = list(zip(pairs, labels, classes))
    random.shuffle(combined)
    pairs, labels, classes = zip(*combined)
    return list(pairs), list(labels), list(classes)



class SiamesePairDataset(Dataset):
    def __init__(self, pairs, labels):
        """
        pairs: list of (item1, item2) (e.g., torch tensors or numpy arrays)
        labels: list of 0/1
        """
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        label = self.labels[idx]
        # Convert label to tensor if not already
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.float32)
        # Ensure x1, x2 are torch tensors
        if not torch.is_tensor(x1):
            x1 = torch.tensor(x1, dtype=torch.float32)
        if not torch.is_tensor(x2):
            x2 = torch.tensor(x2, dtype=torch.float32)
        return x1, x2, label



def view_pairs(pairs, labels, classes=None, num_to_show=5, cmap='magma'):
    """
    Visualize pairs of data (e.g., spectrograms, images) side by side.
    Args:
        pairs: list of (item1, item2) pairs (each item: torch.Tensor or np.ndarray)
        labels: list of 0 (positive) or 1 (negative)
        classes: list of (class1, class2) tuples or None
        num_to_show: number of pairs to display
        cmap: matplotlib colormap
    """
    num_to_show = min(num_to_show, len(pairs))
    for i in range(num_to_show):
        a, b = pairs[i]
        label = labels[i]
        cls = classes[i] if classes is not None else ('?', '?')

        # Convert to numpy for plotting
        def to_np(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().squeeze()
                return x.numpy()
            elif isinstance(x, np.ndarray):
                return np.squeeze(x)
            else:
                raise ValueError("Unsupported data type for visualization.")

        a_np = to_np(a)
        b_np = to_np(b)

        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(a_np, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"{cls[0]}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(b_np, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"{cls[1]}")
        plt.axis('off')

        pair_type = "Positive (0)" if label == 0 else "Negative (1)"
        plt.suptitle(f"Pair {i+1} - {pair_type}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()