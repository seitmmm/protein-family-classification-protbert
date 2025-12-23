# mutations_web.py
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def mutate_sequence_random(seq: str, k: int, rng: np.random.Generator) -> str:
    """
    Случайно мутируем k позиций в последовательности.
    На каждой выбранной позиции ставим случайную аминокислоту,
    отличную от исходной.
    """
    L = len(seq)
    if L == 0:
        return seq

    # ограничиваем k разумными пределами
    k = max(1, min(k, L))

    seq_list = list(seq)
    positions = rng.choice(L, size=k, replace=False)

    for pos in positions:
        orig = seq_list[pos]
        # кандидаты = все аминокислоты кроме исходной
        candidates = [aa for aa in AMINO_ACIDS if aa != orig]
        seq_list[pos] = rng.choice(candidates)

    return "".join(seq_list)


def mutation_stress_test(
    seq: str,
    k: int,
    n_mutants: int,
    base_family: str,
    embedder,
    predict_fn,
):
    """
    Стресс-тест устойчивости к мутациям.

    seq         : исходная аминокислотная последовательность
    k           : число мутируемых позиций (уже абсолютное число!)
    n_mutants   : сколько мутантов сгенерировать
    base_family : семейство, которое считаем "правильным"
    embedder    : ProtBertEmbedder (или другой эмбеддер)
    predict_fn  : функция, принимающая embedding (np.ndarray)
                  и возвращающая dict как predict_with_mode:

                  {
                    "predicted_family": str,
                    "raw_probs": {family: prob, ...},
                    ...
                  }

    Возвращает dict:
      {
        "k": k,
        "n_mutants": n_mutants,
        "accuracy": float,   # доля предсказаний = base_family
        "mean_prob": float,  # средняя P(base_family) по мутантам
      }
    """
    rng = np.random.default_rng()
    n_correct = 0
    probs_true = []

    for _ in range(n_mutants):
        mut_seq = mutate_sequence_random(seq, k, rng)
        emb_mut = embedder.get_embedding(mut_seq)
        res = predict_fn(emb_mut)

        pred_family = res["predicted_family"]
        if pred_family == base_family:
            n_correct += 1

        prob_true = res["raw_probs"].get(base_family, 0.0)
        probs_true.append(prob_true)

    acc = n_correct / n_mutants if n_mutants > 0 else 0.0
    mean_prob = float(np.mean(probs_true)) if probs_true else 0.0

    return {
        "k": k,
        "n_mutants": n_mutants,
        "n_correct": n_correct,
        "accuracy": acc,
        "mean_prob": mean_prob,
    }
