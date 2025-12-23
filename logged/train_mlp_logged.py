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
models_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(models_dir)
emb_dir = os.path.join(project_root, "embeddings")
proc_dir = os.path.join(project_root, "data", "processed")
logs_dir = os.path.join(project_root, "logged", "logs")

# === dataset ===
class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# === MLP model ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def plot_confusion(y_true, y_pred, id_to_family, title="MLP Confusion Matrix"):
    classes = sorted(list(set(y_true)))
    class_names = [id_to_family[c] for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    # ==== load data ====
    X = np.load(os.path.join(emb_dir, "X.npy"))
    y = np.load(os.path.join(emb_dir, "y.npy"))

    with open(os.path.join(proc_dir, "label_map.json"), "r") as f:
        family_to_id = json.load(f)
    id_to_family = {v: k for k, v in family_to_id.items()}

    num_classes = len(family_to_id)
    input_dim = X.shape[1]

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Num classes: {num_classes}, input dim: {input_dim}")

    # ==== split: train / val / test ====
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
    )
    # итого: 70% train, 10% val, 20% test

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # ==== scaling ====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ==== datasets / loaders ====
    train_ds = ProteinDataset(X_train, y_train)
    val_ds = ProteinDataset(X_val, y_val)
    test_ds = ProteinDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ==== model, loss, optimizer ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ==== training loop with early stopping ====
    num_epochs = 50
    best_val_loss = float("inf")
    best_state = None
    patience = 7
    no_improve_epochs = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, num_epochs + 1):
        # --- train ---
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

        # --- validation ---
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

        # early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

    # load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(logs_dir, exist_ok=True)
    np.save(os.path.join(logs_dir, "mlp_train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(logs_dir, "mlp_val_loss.npy"), np.array(val_losses))
    np.save(os.path.join(logs_dir, "mlp_train_acc.npy"), np.array(train_accs))
    np.save(os.path.join(logs_dir, "mlp_val_acc.npy"), np.array(val_accs))
    print(f"Saved training logs to {logs_dir}")

    # ==== evaluation on test set ====
    model.eval()
    y_true = []
    y_pred = []

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
    print(f"\n=== MLP TEST ACCURACY: {test_acc * 100:.2f}% ===")

    # === per-class accuracy ===
    print("\nPer-class accuracy (MLP):")
    classes = sorted(list(set(y_true)))
    for cls in classes:
        mask = (y_true == cls)
        cls_acc = (y_pred[mask] == y_true[mask]).mean()
        print(f"  {id_to_family[cls]}: {cls_acc * 100:.2f}%  (n={mask.sum()})")

    # confusion matrix
    plot_confusion(y_true, y_pred, id_to_family, title="MLP Confusion Matrix")


if __name__ == "__main__":
    main()