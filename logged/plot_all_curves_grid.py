import os
import numpy as np
import matplotlib.pyplot as plt

# директории
logged_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(logged_dir)
logs_dir = os.path.join(project_root, "logged", "logs")
fig_dir = os.path.join(project_root, "logged", "figures")
os.makedirs(fig_dir, exist_ok=True)


def load_val_acc(prefix):
    path = os.path.join(logs_dir, f"{prefix}_val_acc.npy")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found")
        return None
    return np.load(path)


def main():
    curves = {
        "MLP v2": "mlp_v2",
        "KAN (ReLU)": "kan",
        "Smooth KAN": "kan_smooth",
    }

    plt.figure(figsize=(8, 5))

    for label, prefix in curves.items():
        val_acc = load_val_acc(prefix)
        if val_acc is None:
            continue
        epochs = np.arange(1, len(val_acc) + 1)
        plt.plot(epochs, val_acc * 100.0, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Validation accuracy curves: MLP v2 vs KAN vs Smooth KAN")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(fig_dir, "val_accuracy_all_models.png")
    plt.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")
    # plt.show()  # если хочешь посмотреть в интерактиве


if __name__ == "__main__":
    main()
