import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
import umap


# --------- конфигурация путей ---------
project_root = os.path.dirname(os.path.abspath(__file__))
emb_dir = os.path.join(project_root, "embeddings")
out_dir = os.path.join(project_root, "analysis_umap")
os.makedirs(out_dir, exist_ok=True)

label_names = [
    "gpcr",
    "ion_channel",
    "abc_transporter",
    "kinase",
    "cytochrome_p450",
    "immunoglobulin",
    "ribosomal_protein",
    "zinc_finger",
]

# --------- загрузка данных ---------
print("Loading embeddings...")
X = np.load(os.path.join(emb_dir, "X.npy"))   # (N, 1024)
y = np.load(os.path.join(emb_dir, "y.npy"))   # (N,)

num_classes = len(label_names)
print(f"X shape: {X.shape}, num_classes: {num_classes}")

# --------- сабсемплинг (как в t-SNE) ---------
N = X.shape[0]
max_points = 25000

if N > max_points:
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

    X_umap_inp = np.concatenate(X_sub_list, axis=0)
    y_umap = np.concatenate(y_sub_list, axis=0)
else:
    X_umap_inp = X
    y_umap = y

print(f"Using {X_umap_inp.shape[0]} points for UMAP")


# --------- UMAP 2D ---------
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42,
)

print("Fitting UMAP...")
X_umap = reducer.fit_transform(X_umap_inp)  # (M, 2)

# --- UMAP: все семейства ---
plt.figure(figsize=(7, 6))
cmap = plt.colormaps.get_cmap("tab10")

for c in range(num_classes):
    mask = (y_umap == c)
    plt.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        s=8,
        alpha=0.7,
        color=cmap(c),
        label=label_names[c],
    )

plt.legend(
    markerscale=2,
    fontsize=8,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.title("UMAP of ProtBERT embeddings (all families)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
path_all = os.path.join(out_dir, "umap_all_families25k.png")
plt.savefig(path_all, dpi=300)
plt.close()
print("Saved:", path_all)

# --- UMAP: подсветка kinase ---
kinase_id = label_names.index("kinase")

plt.figure(figsize=(7, 6))
# все серым
plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    s=6,
    alpha=0.15,
    color="lightgray",
    label="other families",
)

# kinase ярко
mask_kin = (y_umap == kinase_id)
plt.scatter(
    X_umap[mask_kin, 0],
    X_umap[mask_kin, 1],
    s=10,
    alpha=0.9,
    color="red",
    label="kinase",
)

plt.legend(fontsize=9)
plt.title("UMAP of ProtBERT embeddings (kinase highlighted)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
path_kin = os.path.join(out_dir, "umap_kinase_highlight25k.png")
plt.savefig(path_kin, dpi=300)
plt.close()
print("Saved:", path_kin)


# --------- Heatmap попарных расстояний между семействами ---------
print("Computing class centroids and distances...")
centroids = []
for c in range(num_classes):
    mask = (y == c)
    mu = X[mask].mean(axis=0)  # средний вектор в исходном 1024D пространстве
    centroids.append(mu)
centroids = np.vstack(centroids)  # (8, 1024)

# евклидовы расстояния между центроидами
dist_matrix = pairwise_distances(centroids, metric="euclidean")

plt.figure(figsize=(6, 5))
sns.heatmap(
    dist_matrix,
    xticklabels=label_names,
    yticklabels=label_names,
    cmap="viridis",
    annot=True,
    fmt=".1f",
)
plt.title("Pairwise distances between family centroids (ProtBERT space)")
plt.tight_layout()
path_heat = os.path.join(out_dir, "family_distance_heatmap25k.png")
plt.savefig(path_heat, dpi=300)
plt.close()
print("Saved:", path_heat)

print("\nDone.")
