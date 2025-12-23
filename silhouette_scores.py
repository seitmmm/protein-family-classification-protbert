import numpy as np
import os
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
emb_dir = os.path.join(project_root, "embeddings")  # путь как в tsne_protbert.py
save_dir = os.path.join(project_root, "analysis_silhouette")
os.makedirs(save_dir, exist_ok=True)

label_names = [
    "gpcr",
    "ion_channel",
    "abc_transporter",
    "kinase",
    "cytochrome_p450",
    "immunoglobulin",
    "ribosomal_protein",
    "zinc_finger"
]

print("Loading embeddings...")
X = np.load(os.path.join(emb_dir, "X.npy"))
y = np.load(os.path.join(emb_dir, "y.npy"))

print("Calculating silhouette scores...")
# Общий silhouette score
overall_score = silhouette_score(X, y)
print(f"\nOverall silhouette score: {overall_score:.4f}")

# Индивидуальные silhouette values для каждого объекта
s_vals = silhouette_samples(X, y)

# Считаем средний score для каждого класса
class_scores = []
for class_id in range(len(label_names)):
    mask = (y == class_id)
    score = s_vals[mask].mean()
    class_scores.append(score)

# Печать таблицы
print("\nSilhouette score per class:")
for name, score in zip(label_names, class_scores):
    print(f"  {name:20s}  {score:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.barh(label_names, class_scores, color="skyblue")
plt.xlabel("Silhouette score")
plt.title("Silhouette score per protein family")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

save_path = os.path.join(save_dir, "silhouette_scores.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\nSaved plot to: {save_path}")
