import os
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Настройки путей и имён файлов
# ===========================

project_root = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(project_root, "logs_mutations")

# Твои файлы:
file_kinase = os.path.join(logs_dir, "mutation_meta_kinase.npz")
file_gpcr   = os.path.join(logs_dir, "mutation_meta_gpcr.npz")


# ===========================
# Функция загрузки одной серии
# ===========================

def load_series(path):
    """
    Ожидаем .npz со структурой:
      - k_list
      - acc
      - mean_p_true
      - true_family
      - base_p_true
      - n_mutants
      - seq_len
    """
    data = np.load(path, allow_pickle=True)
    k_list      = data["k_list"]
    acc         = data["acc"]
    mean_p_true = data["mean_p_true"]
    family      = str(data["true_family"])
    base_p      = float(data["base_p_true"])
    n_mutants   = int(data["n_mutants"])
    seq_len     = int(data["seq_len"])
    return {
        "k_list": k_list,
        "acc": acc,
        "mean_p_true": mean_p_true,
        "family": family,
        "base_p_true": base_p,
        "n_mutants": n_mutants,
        "seq_len": seq_len,
    }


def main():
    # --- загружаем обе серии ---
    kin = load_series(file_kinase)
    gp  = load_series(file_gpcr)

    print("Loaded:")
    print("  kinase from:", file_kinase)
    print("  gpcr   from:", file_gpcr)

    # Проверка, что k_list совпадает (если нет — всё равно отрисуем)
    if np.array_equal(kin["k_list"], gp["k_list"]):
        k_list = kin["k_list"]
    else:
        print("WARNING: k_list differ between files; will plot them as-is.")
        k_list = kin["k_list"]  # но ниже будем использовать свои для каждой серии

    # ===========================
    # 1. Accuracy vs k
    # ===========================
    plt.figure(figsize=(8, 5))
    plt.plot(kin["k_list"], kin["acc"], marker="o", label=f"{kin['family']} (acc)")
    plt.plot(gp["k_list"],  gp["acc"],  marker="o", label=f"{gp['family']} (acc)")

    plt.xlabel("Number of mutations (k)")
    plt.ylabel("Accuracy on mutants")
    plt.title("Mutation robustness: accuracy vs number of mutations")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    acc_plot_path = os.path.join(logs_dir, "mutation_compare_accuracy.png")
    plt.savefig(acc_plot_path, dpi=300)
    print("Saved accuracy plot to:", acc_plot_path)

    # ===========================
    # 2. Mean P(true_family) vs k
    # ===========================
    plt.figure(figsize=(8, 5))
    plt.plot(kin["k_list"], kin["mean_p_true"], marker="o",
             label=f"{kin['family']} (mean P)")
    plt.plot(gp["k_list"],  gp["mean_p_true"],  marker="o",
             label=f"{gp['family']} (mean P)")

    plt.xlabel("Number of mutations (k)")
    plt.ylabel("Mean P(true family)")
    plt.title("Mutation robustness: mean confidence vs number of mutations")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    conf_plot_path = os.path.join(logs_dir, "mutation_compare_confidence.png")
    plt.savefig(conf_plot_path, dpi=300)
    print("Saved confidence plot to:", conf_plot_path)

    plt.show()


if __name__ == "__main__":
    main()
