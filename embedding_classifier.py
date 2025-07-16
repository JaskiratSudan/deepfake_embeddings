import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.decomposition import PCA

def train_svm_classifier(X, y, kernel='rbf', C=1.0, gamma='scale', test_size=0.2, random_state=42):
    """
    Trains an SVM classifier and returns the trained model and test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, plot_pca=True):
    """
    Evaluates a trained model and plots confusion matrix, ROC curve, and optionally 2D PCA projection.

    Parameters:
    - model: trained pipeline (with scaler and SVC)
    - X_test: test features
    - y_test: test labels
    - plot_pca: whether to plot a 2D PCA projection of the test data
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # PCA projection (2D)
    if plot_pca:
        X_scaled = model.named_steps['standardscaler'].transform(X_test)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', alpha=0.7)
        plt.title("2D PCA Projection")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(*scatter.legend_elements(), title="Class")

    plt.tight_layout()
    plt.show()


def plot_tsne_with_svm_2d(model, embeddings, labels, title="2D t-SNE with SVM", draw_boundary=True):
    """
    Visualize 128D embeddings reduced to 2D with t-SNE, show support vectors, and optionally draw decision boundary.

    Parameters:
    - model: trained SVM pipeline (StandardScaler + SVC)
    - embeddings: original input data (not scaled)
    - labels: list or array of original class labels (e.g., 'real', 'fake')
    - draw_boundary: whether to draw an approximate decision boundary in t-SNE space
    """
    # Extract scaler and SVC from pipeline
    scaler = model.named_steps['standardscaler']
    svc = model.named_steps['svc']

    # Scale embeddings to match what SVC saw
    X_scaled = scaler.transform(embeddings)
    
    # Get support vectors' indices
    support_indices = svc.support_

    # Reduce original embeddings (not scaled) to 2D
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)

    # Map support vectors to t-SNE space
    support_mask = np.zeros(len(embeddings), dtype=bool)
    support_mask[support_indices] = True

    # Plot
    plt.figure(figsize=(8, 6))
    label_set = sorted(set(labels))
    colors = ['blue', 'red', 'green', 'orange']
    color_map = {l: c for l, c in zip(label_set, colors)}

    for l in label_set:
        idx = [i for i, lab in enumerate(labels) if lab == l and not support_mask[i]]
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=l, color=color_map[l], alpha=0.5, edgecolor='k', s=40)

    # Plot support vectors with black edge
    for l in label_set:
        idx = [i for i, lab in enumerate(labels) if lab == l and support_mask[i]]
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f"{l} (SV)", 
                    color=color_map[l], alpha=1, edgecolor='black', linewidth=1.2, s=70, marker='o')

    # Optional: Approximate decision boundary using t-SNE space
    if draw_boundary:
        try:
            from sklearn.neighbors import KNeighborsClassifier
            # Train a KNN classifier on t-SNE 2D to approximate decision zones
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_tsne, labels)

            # Create mesh grid
            x_min, x_max = X_tsne[:, 0].min() - 5, X_tsne[:, 0].max() + 5
            y_min, y_max = X_tsne[:, 1].min() - 5, X_tsne[:, 1].max() + 5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = knn.predict(grid)
            Z = np.array(Z).reshape(xx.shape)

            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([color_map[l] for l in label_set])
            plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.1)
        except:
            print("⚠️ Could not draw boundary. Make sure all labels are numeric or strings.")

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.show()