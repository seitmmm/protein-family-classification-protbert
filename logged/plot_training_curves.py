import os
import numpy as np
import matplotlib.pyplot as plt

# пути
logged_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(logged_dir)
logs_dir = os.path.join(project_root, "logged", "logs")
fig_dir = os.path.join(project_root, "logged", "figures")
os.makedirs(fig_dir, exist_ok=True)


def load_curves(prefix):
    """Загружает train/val loss/acc для модели по префиксу (mlp, mlp_v2, kan, kan_smooth)."""
    def load_one(name):
        path = os.path.join(logs_dir, f"{prefix}_{name}.npy")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found")
            return None
        return np.load(path)

    train_loss = load_one("train_loss")
    val_loss = load_one("val_loss")
    train_acc = load_one("train_acc")
    val_acc = load_one("val_acc")
    return train_loss, val_loss, train_acc, val_acc


def plot_model_curves(prefix, title):
    train_loss, val_loss, train_acc, val_acc = load_curves(prefix)
    if train_loss is None or val_loss is None or train_acc is None or val_acc is None:
        print(f"[SKIP] Not all logs for {prefix} present")
        return

    epochs_loss = np.arange(1, len(train_loss) + 1)
    epochs_acc = np.arange(1, len(train_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Loss ---
    axes[0].plot(epochs_loss, train_loss, label="Train loss")
    axes[0].plot(epochs_loss, val_loss, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} – Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs_acc, np.array(train_acc) * 100, label="Train acc")
    axes[1].plot(epochs_acc, np.array(val_acc) * 100, label="Val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{title} – Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(fig_dir, f"{prefix}_training_curves.png")
    fig.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")

    # если хочешь, можно включить/выключить show
    # plt.show()
    plt.close(fig)


def main():
    models = [
        ("mlp", "MLP (baseline)"),
        ("mlp_v2", "MLP v2"),
        ("kan", "KAN (ReLU splines)"),
        ("kan_smooth", "Smooth KAN (Gaussian splines)"),
    ]

    for prefix, title in models:
        print(f"=== Plotting curves for {title} ({prefix}) ===")
        plot_model_curves(prefix, title)


if __name__ == "__main__":
    main()
