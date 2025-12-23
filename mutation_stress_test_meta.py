import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

# ================================
# Настройка путей
# ================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from utils.embedder import ProtBertEmbedder
from models.inference_ensemble import load_ensemble, predict_from_embedding


# ================================
# Функции для мутаций
# ================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    seq = "".join(seq.split())
    return seq


def random_point_mutation(seq: str):
    seq = clean_sequence(seq)
    if not seq:
        raise ValueError("Empty sequence for mutation.")

    L = len(seq)
    pos = random.randint(0, L - 1)
    orig_aa = seq[pos]

    candidates = [aa for aa in AMINO_ACIDS if aa != orig_aa]
    if not candidates:
        return seq, pos, orig_aa, orig_aa

    new_aa = random.choice(candidates)
    mutated_seq = seq[:pos] + new_aa + seq[pos + 1:]

    return mutated_seq, pos, orig_aa, new_aa


def random_k_mutations(seq: str, k: int):
    mutated = clean_sequence(seq)
    info = []

    for _ in range(k):
        mutated, pos, orig_aa, new_aa = random_point_mutation(mutated)
        info.append((pos, orig_aa, new_aa))

    return mutated, info


def generate_mutant_panel(seq: str, num_mutations_list, mutants_per_level=50, seed=42):
    random.seed(seed)
    panel = {}

    for k in num_mutations_list:
        variants = []
        for _ in range(mutants_per_level):
            mut_seq, info = random_k_mutations(seq, k)
            variants.append((mut_seq, info))
        panel[k] = variants

    return panel


# ================================
# Основной эксперимент
# ================================

def main():
    # ===== 0. Вставляешь сюда любую последовательность =====
    base_sequence = """
MPRVKAAQAGRQSSAKRHLAEQFAVGEIITDMAKKEWKVGLPIGQGGFGCIYLADMNSSE
SVGSDAPCVVKVEPSDNGPLFTELKFYQRAAKPEQIQKWIRTRKLKYLGVPKYWGSGLHD
KNGKSYRFMIMDRFGSDLQKIYEANAKRFSRKTVLQLSLRILDILEYIHEHEYVHGDIKA
SNLLLNYKNPDQVYLVDYGLAYRYCPEGVHKEYKEDPKRCHDGTIEFTSIDAHNGVAPSR
RGDLEILGYCMIQWLTGHLPWEDNLKDPKYVRDSKIRYRENIASLMDKCFPEKNKPGEIA
KYMETVKLLDYTEKPLYENLRDILLQGLKAIGSKDDGKLDLSVVENGGLKAKTITKKRKK
EIEESKEPGVEDTEWSNTQTEEAIQTRSRTRKRVQK
"""
    # Истинное семейство
    true_family_name = "kinase"

    # Сколько мутаций будем пробовать
    num_mutations_list = [1, 3, 5, 10, 50, 100]
    mutants_per_level = 50

    # Куда сохранять графики
    out_dir = os.path.join(PROJECT_ROOT, "logs_mutations")
    os.makedirs(out_dir, exist_ok=True)

    # ===== 1. Модели и embedder =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embedder = ProtBertEmbedder(device=device)
    ensemble = load_ensemble()

    family_to_id = ensemble["family_to_id"]
    id_to_family = ensemble["id_to_family"]

    if true_family_name not in family_to_id:
        raise ValueError(
            f"Unknown family '{true_family_name}'. "
            f"Available: {list(family_to_id.keys())}"
        )

    base_seq_clean = clean_sequence(base_sequence)
    print(f"\nBase sequence length: {len(base_seq_clean)} aa")

    # ===== 2. Базовое предсказание =====
    print("\n=== BASE SEQUENCE PREDICTION (META-ENSEMBLE) ===")
    base_emb = embedder.get_embedding(base_seq_clean)

    base_res = predict_from_embedding(
        base_emb,
        ensemble,
        topk=5,
        threshold=0.0,
    )

    print(f"Predicted family: {base_res['predicted_family']}")
    print("Top-5:")
    for fam, prob in base_res["topk"]:
        print(f"  {fam:20s} : {prob:.3f}")

    base_prob_true = base_res["raw_probs"].get(true_family_name, 0.0)
    print(f"\nP(true family = {true_family_name}) for base sequence: {base_prob_true:.3f}")

    # ===== 3. Генерируем мутантов =====
    print("\n=== GENERATING MUTANTS ===")
    panel = generate_mutant_panel(
        base_seq_clean,
        num_mutations_list=num_mutations_list,
        mutants_per_level=mutants_per_level,
        seed=42,
    )

    for k in num_mutations_list:
        print(f"  k = {k}: {len(panel[k])} mutants")

    # ===== 4. Прогоняем через meta-ensemble =====
    print("\n=== EVALUATING MUTATION ROBUSTNESS (META-ENSEMBLE) ===")

    from collections import defaultdict
    stats_correct = defaultdict(int)
    stats_total = defaultdict(int)
    stats_prob_true = defaultdict(list)

    for k in num_mutations_list:
        variants = panel[k]
        for mut_seq, mut_info in variants:
            emb = embedder.get_embedding(mut_seq)

            res = predict_from_embedding(
                emb,
                ensemble,
                topk=3,
                threshold=0.0
            )

            pred_family = res["predicted_family"]
            prob_true = res["raw_probs"].get(true_family_name, 0.0)

            stats_total[k] += 1
            if pred_family == true_family_name:
                stats_correct[k] += 1

            stats_prob_true[k].append(prob_true)

    # ===== 5. Числа + подготовка массивов =====
    print("\n=== MUTATION ROBUSTNESS SUMMARY (META-ENSEMBLE) ===")
    print(f"True family: {true_family_name}\n")

    k_values = []
    acc_values = []
    mean_conf_values = []

    for k in num_mutations_list:
        total = stats_total[k]
        correct = stats_correct[k]
        acc = correct / total if total > 0 else 0.0
        mean_prob = float(np.mean(stats_prob_true[k])) if stats_prob_true[k] else 0.0

        print(f"k = {k:3d} mutations:")
        print(f"  Correct predictions: {correct}/{total}  (acc = {acc*100:.2f}%)")
        print(f"  Mean P(true_family) over mutants: {mean_prob:.3f}")
        print()

        k_values.append(k)
        acc_values.append(acc)
        mean_conf_values.append(mean_prob)

    # ===== 6. Графики =====

    # 6.1 Accuracy vs k
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, acc_values, marker="o")
    plt.xlabel("Number of mutations (k)")
    plt.ylabel("Accuracy on mutants")
    plt.title(f"Mutation robustness (family = {true_family_name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, f"{true_family_name}_mutation_acc_vs_k.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print("Saved:", acc_path)

    # 6.2 Mean confidence vs k
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, mean_conf_values, marker="o")
    plt.xlabel("Number of mutations (k)")
    plt.ylabel(f"Mean P({true_family_name})")
    plt.title(f"Mean confidence vs mutations (family = {true_family_name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    conf_path = os.path.join(out_dir, f"{true_family_name}_mutation_conf_vs_k.png")
    plt.savefig(conf_path, dpi=300)
    plt.close()
    print("Saved:", conf_path)

    # 6.3 Boxplot распределений P(true_family) по k
    plt.figure(figsize=(7, 4))
    data = [stats_prob_true[k] for k in num_mutations_list]
    plt.boxplot(data, positions=range(len(num_mutations_list)))
    plt.xticks(range(len(num_mutations_list)), [str(k) for k in num_mutations_list])
    plt.xlabel("Number of mutations (k)")
    plt.ylabel(f"P({true_family_name})")
    plt.title(f"Distribution of P({true_family_name}) over mutants")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    box_path = os.path.join(out_dir, f"{true_family_name}_mutation_boxplot_conf_vs_k.png")
    plt.savefig(box_path, dpi=300)
    plt.close()
    print("Saved:", box_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
