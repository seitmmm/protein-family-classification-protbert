import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# пути (как в твоих моделях)
project_root = os.path.dirname(os.path.abspath(__file__))
emb_dir = os.path.join(project_root, "embeddings")
proc_dir = os.path.join(project_root, "data", "processed")
plots_dir = os.path.join(project_root, "logs_tsne")
os.makedirs(plots_dir, exist_ok=True)

# ===== 1. загрузка X, y и маппинга =====
X = np.load(os.path.join(emb_dir, "X.npy"))   # (N, 1024)
y = np.load(os.path.join(emb_dir, "y.npy"))   # (N,)

with open(os.path.join(proc_dir, "label_map.json"), "r") as f:
    family_to_id = json.load(f)
id_to_family = {v: k for k, v in family_to_id.items()}

num_classes = len(family_to_id)
print(f"X shape: {X.shape}, num classes: {num_classes}")

# хотим разумный размер для t-SNE: можно взять всё, но если N очень большое,
# можно ограничить, например, 8000–10000 точками
N = X.shape[0]
max_points = 25000
if N > max_points:
    # стратифицированный сабсемплинг по классам
    X_sub_list = []
    y_sub_list = []
    per_class = max_points // num_classes

    rng = np.random.default_rng(42)
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) <= per_class:
            chosen = idx_c
        else:
            chosen = rng.choice(idx_c, size=per_class, replace=False)
        X_sub_list.append(X[chosen])
        y_sub_list.append(y[chosen])

    X_tsne = np.concatenate(X_sub_list, axis=0)
    y_tsne = np.concatenate(y_sub_list, axis=0)
else:
    X_tsne = X
    y_tsne = y

print(f"Using {X_tsne.shape[0]} points for t-SNE")

# ===== 2. t-SNE в 2D =====
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
)
X_2d = tsne.fit_transform(X_tsne)   # (M, 2)

# ===== 3. Общий plot: все классы в разных цветах =====
plt.figure(figsize=(7, 6))

colors = plt.cm.get_cmap("tab10", num_classes)

for c in range(num_classes):
    mask = (y_tsne == c)
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        s=8,
        alpha=0.7,
        color=colors(c),
        label=id_to_family[c],
    )

plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("t-SNE of ProtBERT embeddings (all families)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "tsne_all_families.png"), dpi=300)
plt.close()

# ===== 4. t-SNE с подсветкой КАЖДОГО семейства =====
for fam_id, fam_name in id_to_family.items():
    plt.figure(figsize=(7, 6))

    # все серым
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        s=6,
        alpha=0.15,
        color="lightgray",
        label="other families",
    )

    # текущее семейство – ярким цветом
    mask = (y_tsne == fam_id)
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        s=10,
        alpha=0.9,
        label=fam_name,
    )

    plt.legend(fontsize=9)
    plt.title(f"t-SNE of ProtBERT embeddings ({fam_name} highlighted)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    fname = f"tsne_highlight_{fam_name}.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=300)
    plt.close()

print("Saved per-family t-SNE highlight plots to:", plots_dir)

