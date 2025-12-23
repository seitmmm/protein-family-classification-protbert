import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# === paths ===
logged_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(logged_dir)
emb_dir = os.path.join(project_root, "embeddings")
proc_dir = os.path.join(project_root, "data", "processed")
logs_dir = os.path.join(project_root, "logged", "logs")


class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SplineActivation(nn.Module):
    def __init__(self, hidden_dim: int, num_knots: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_knots = num_knots
        self.knots = nn.Parameter(torch.linspace(-2, 2, num_knots).repeat(hidden_dim, 1))
        self.coeffs = nn.Parameter(torch.randn(hidden_dim, num_knots) * 0.1)

    def forward(self, z):
        z_expanded = z.unsqueeze(-1)
        knots = self.knots.unsqueeze(0)
        coeffs = self.coeffs.unsqueeze(0)
        relu_terms = torch.relu(z_expanded - knots)
        out = (coeffs * relu_terms).sum(dim=-1)
        return out


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 5, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.spline = SplineActivation(out_dim, num_knots=num_knots)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.linear(x)
        z = self.spline(z)
        z = self.bn(z)
        z = torch.relu(z)
        z = self.dropout(z)
        return z


class KANClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_knots: int = 5):
        super().__init__()
        self.layer1 = KANLayer(input_dim, 768, num_knots=num_knots, dropout=0.2)
        self.layer2 = KANLayer(768, 512, num_knots=num_knots, dropout=0.2)
        self.layer3 = KANLayer(512, 256, num_knots=num_knots, dropout=0.2)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out(x)
        return x


def plot_confusion(
    y_true,
    y_pred,
    id_to_family,
    title="KAN (ReLU) Confusion Matrix",
    normalize=True,
    save_path=None,
):

    # фиксируем порядок классов: 0..num_classes-1
    classes = sorted(list(set(y_true)))
    class_names = [id_to_family[c] for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        # нормализация по строкам (True class)
        cm = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=False,          # можно поставить True, но будет мелко
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_confusion_paper(
            y_true,
            y_pred,
            id_to_family,
            title="KAN (ReLU) Confusion Matrix",
            normalize=True,
            save_path=None,
    ):
        classes = sorted(list(set(y_true)))
        class_names = [id_to_family[c] for c in classes]

        cm = confusion_matrix(y_true, y_pred, labels=classes)

        if normalize:
            cm = cm.astype("float")
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        plt.figure(figsize=(5, 4))
        # максимально похоже на их картинку:
        im = plt.imshow(
            cm,
            interpolation="nearest",
            cmap="jet",  # синий → зелёный → жёлтый → красный
            vmin=0.0 if normalize else None,
            vmax=1.0 if normalize else None,
            aspect="auto",
        )
        plt.title(title)
        plt.xlabel("Predicted families")
        plt.ylabel("True families")

        # тики – наши 8 классов (у них 589, поэтому у них визуально сплошная диагональ)
        tick_positions = np.arange(len(class_names))
        plt.xticks(tick_positions, class_names, rotation=45, ha="right", fontsize=8)
        plt.yticks(tick_positions, class_names, fontsize=8)

        cbar = plt.colorbar(im)
        cbar.set_label("Normalized frequency" if normalize else "Count")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


def main():
    X = np.load(os.path.join(emb_dir, "X.npy"))
    y = np.load(os.path.join(emb_dir, "y.npy"))

    with open(os.path.join(proc_dir, "label_map.json"), "r") as f:
        family_to_id = json.load(f)
    id_to_family = {v: k for k, v in family_to_id.items()}

    num_classes = len(family_to_id)
    input_dim = X.shape[1]

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Num classes: {num_classes}, input dim: {input_dim}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_ds = ProteinDataset(X_train, y_train)
    val_ds = ProteinDataset(X_val, y_val)
    test_ds = ProteinDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = KANClassifier(input_dim=input_dim, num_classes=num_classes, num_knots=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    num_epochs = 80
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    no_improve_epochs = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= total
        val_acc = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(logs_dir, exist_ok=True)
    np.save(os.path.join(logs_dir, "kan_train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(logs_dir, "kan_val_loss.npy"),   np.array(val_losses))
    np.save(os.path.join(logs_dir, "kan_train_acc.npy"),  np.array(train_accs))
    np.save(os.path.join(logs_dir, "kan_val_acc.npy"),    np.array(val_accs))
    print(f"Saved KAN logs to {logs_dir}")

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            _, preds = torch.max(logits, 1)

            y_true.extend(y_batch.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_acc = accuracy_score(y_true, y_pred)
    print(f"\n=== KAN TEST ACCURACY: {test_acc * 100:.2f}% ===")

    print("\nPer-class accuracy (KAN):")
    classes = sorted(list(set(y_true)))
    for cls in classes:
        mask = (y_true == cls)
        cls_acc = (y_pred[mask] == y_true[mask]).mean()
        print(f"  {id_to_family[cls]}: {cls_acc * 100:.2f}%  (n={mask.sum()})")

    plot_confusion(
        y_true,
        y_pred,
        id_to_family,
        title="KAN (ReLU) Confusion Matrix (normalized)",
        normalize=True,
        save_path=os.path.join(logs_dir, "kan_relu_confusion_norm.png"),
    )

    plot_confusion_paper(
        y_true,
        y_pred,
        id_to_family,
        title="KAN (ReLU) Confusion Matrix",
        normalize=True,
        save_path=os.path.join(logs_dir, "kan_relu_confusion_paper.png"),
    )


if __name__ == "__main__":
    main()
