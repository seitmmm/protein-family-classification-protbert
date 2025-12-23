import os
import sys
import random
import numpy as np

import torch  # нужен, чтобы embedder и модели нормально работали

# --- путь к проекту, чтобы видеть utils/ и models/ ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.embedder import ProtBertEmbedder
from models.inference_ensemble import load_ensemble, predict_from_embedding


# ==============================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ==============================

# 1) сюда вставляешь последовательность интересующего белка
BASE_SEQ = """
MLAAAFADSNSSSMNVSFAHLHFAGGYLPSDSQDWRTIIPALLVAVCLVGFVGNLCVIGI
LLHNAWKGKPSMIHSLILNLSLADLSLLLFSAPIRATAYSKSVWDLGWFVCKSSDWFIHT
CMAAKSLTIVVVAKVCFMYASDPAKQVSIHNYTIWSVLVAIWTVASLLPLPEWFFSTIRH
HEGVEMCLVDVPAVAEEFMSMFGKLYPLLAFGLPLFFASFYFWRAYDQCKKRGTKTQNLR
NQIRSKQVTVMLLSIAIISALLWLPEWVAWLWVWHLKAAGPAPPQGFIALSQVLMFSISS
ANPLIFLVMSEEFREGLKGVWKWMITKKPPTVSESQETPAGNSEGLPDKVPSPESPASIP
EKEKPSSPSSGKGKTEKAEIPILPDVEQFWHERDTVPSVQDNDPIPWEHEDQETGEGVK
""".replace("\n", "").strip()

# 2) истинное семейство для этой последовательности
TRUE_FAMILY = "gpcr"

# 3) набор k (сколько позиций мутируем) и сколько мутантов на каждое k
K_LIST = [1, 3, 5, 10, 50, 100]
N_MUTANTS_PER_K = 50

# аминокислотный алфавит
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def random_mutation(seq: str, k: int) -> str:
    """
    Делает k случайных точечных мутаций в последовательности.
    В каждой позиции выбирается другая аминокислота.
    """
    seq_list = list(seq)
    L = len(seq_list)
    k = min(k, L)  # на всякий случай

    positions = random.sample(range(L), k)
    for pos in positions:
        orig = seq_list[pos]
        choices = [aa for aa in AA_ALPHABET if aa != orig]
        seq_list[pos] = random.choice(choices)

    return "".join(seq_list)


def predict_family_and_prob(seq: str, embedder, ensemble, true_family: str):
    """
    Получить:
      - предсказанное семейство (по meta-ensemble)
      - вероятность P(true_family)
    """
    emb = embedder.get_embedding(seq)  # (1024,)
    res = predict_from_embedding(emb, ensemble, topk=5, threshold=0.0)
    pred_family = res["predicted_family"]
    p_true = float(res["raw_probs"].get(true_family, 0.0))
    return pred_family, p_true


# ==============================
# MAIN
# ==============================

def main():
    print(f"Base sequence length: {len(BASE_SEQ)} aa")
    print(f"True family: {TRUE_FAMILY}\n")

    # --- грузим ProtBERT и мета-ансамбль ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embedder = ProtBertEmbedder(device=device)
    ensemble = load_ensemble()

    # --- предсказание для базовой последовательности ---
    base_pred, base_p_true = predict_family_and_prob(
        BASE_SEQ, embedder, ensemble, TRUE_FAMILY
    )

    print("=== BASE SEQUENCE PREDICTION (META-ENSEMBLE) ===")
    print(f"Predicted family: {base_pred}")
    print(f"P(true family = {TRUE_FAMILY}) = {base_p_true:.3f}\n")

    # --- бежим по k и собираем статистику ---
    acc_by_k = {}
    mean_p_by_k = {}

    print("=== EVALUATING MUTATION ROBUSTNESS (META-ENSEMBLE) ===\n")

    for k in K_LIST:
        correct = 0
        probs_true = []

        print(f"Running k = {k} ({N_MUTANTS_PER_K} mutants)...")

        for _ in range(N_MUTANTS_PER_K):
            mut_seq = random_mutation(BASE_SEQ, k)
            pred_family, p_true = predict_family_and_prob(
                mut_seq, embedder, ensemble, TRUE_FAMILY
            )

            if pred_family == TRUE_FAMILY:
                correct += 1
            probs_true.append(p_true)

        acc = correct / N_MUTANTS_PER_K
        mean_p = float(np.mean(probs_true))

        acc_by_k[k] = acc
        mean_p_by_k[k] = mean_p

        print(
            f"k = {k:3d}: "
            f"acc = {acc*100:5.2f}%  |  "
            f"mean P({TRUE_FAMILY}) = {mean_p:.3f}"
        )

    # --- финальное резюме ---
    print("\n=== MUTATION ROBUSTNESS SUMMARY (META-ENSEMBLE) ===")
    print(f"True family: {TRUE_FAMILY}")
    for k in K_LIST:
        print(
            f"k = {k:3d}: "
            f"acc = {acc_by_k[k]*100:5.2f}%, "
            f"mean P({TRUE_FAMILY}) = {mean_p_by_k[k]:.3f}"
        )

    # --- сохраняем результаты на диск ---
    logs_dir = os.path.join(project_root, "logs_mutations")
    os.makedirs(logs_dir, exist_ok=True)

    out_path = os.path.join(
        logs_dir,
        f"mutation_meta_{TRUE_FAMILY}.npz"
    )

    np.savez(
        out_path,
        k_list=np.array(K_LIST, dtype=int),
        acc=np.array([acc_by_k[k] for k in K_LIST], dtype=float),
        mean_p_true=np.array([mean_p_by_k[k] for k in K_LIST], dtype=float),
        true_family=TRUE_FAMILY,
        base_p_true=base_p_true,
        n_mutants=N_MUTANTS_PER_K,
        seq_len=len(BASE_SEQ),
    )

    print(f"\nSaved mutation statistics to:\n  {out_path}")


if __name__ == "__main__":
    main()
