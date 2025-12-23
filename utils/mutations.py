import random
from typing import List, Tuple

# Стандартный набор 20 аминокислот
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def random_point_mutation(seq: str) -> Tuple[str, int, str, str]:
    """
    Одна случайная точечная мутация:
      - выбираем случайную позицию
      - подбираем случайный аминокислотный символ, отличный от исходного.

    Возвращает:
      (mutated_seq, pos, orig_aa, new_aa)
    """
    if not seq:
        raise ValueError("Empty sequence.")

    seq = seq.strip().upper()
    L = len(seq)
    pos = random.randint(0, L - 1)
    orig_aa = seq[pos]

    # возможные замены, исключаем оригинальный символ
    candidates = [aa for aa in AMINO_ACIDS if aa != orig_aa]
    if not candidates:
        # почти нереально, но на всякий случай
        return seq, pos, orig_aa, orig_aa

    new_aa = random.choice(candidates)
    mutated_seq = seq[:pos] + new_aa + seq[pos+1:]

    return mutated_seq, pos, orig_aa, new_aa


def random_k_mutations(seq: str, k: int) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    K мутаций в последовательности.
    Мутации могут попадать в разные / те же позиции (простая версия, без контроля уникальности).

    Возвращает:
      mutated_seq, mutations_info
    где mutations_info = [(pos, orig_aa, new_aa), ...]
    """
    mutated = seq
    info = []

    for _ in range(k):
        mutated, pos, orig_aa, new_aa = _one_mutation_step(mutated)

        info.append((pos, orig_aa, new_aa))

    return mutated, info


def _one_mutation_step(seq: str) -> Tuple[str, int, str, str]:
    """
    Внутренняя функция, чтобы k-мутаций можно было накладывать последовательно.
    """
    mutated_seq, pos, orig_aa, new_aa = random_point_mutation(seq)
    return mutated_seq, pos, orig_aa, new_aa


def generate_mutant_panel(
    seq: str,
    num_mutations_list: List[int],
    mutants_per_level: int = 50,
    seed: int = 42
):
    """
    Генерируем “панель” мутантов для разных уровней мутаций.

    Например:
      num_mutations_list = [1, 2, 5, 10]
      mutants_per_level = 50

    Вернёт словарь:
    {
      1: [ (mut_seq1, info1), (mut_seq2, info2), ... ],
      2: [ ... ],
      ...
    }

    infoN = список (pos, orig_aa, new_aa) для каждой мутации.
    """
    random.seed(seed)
    panel = {}

    for k in num_mutations_list:
        variants = []
        for _ in range(mutants_per_level):
            mut_seq, info = random_k_mutations(seq, k)
            variants.append((mut_seq, info))
        panel[k] = variants

    return panel