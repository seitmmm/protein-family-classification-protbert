import requests


class ESMFoldWrapper:
    """
    Обёртка над ESMFold API (esmatlas.com).
    Вместо локальной тяжёлой модели используем HTTP-запрос:
    На вход: аминокислотная последовательность (строка).
    На выход: PDB как текст.
    """

    def __init__(self, device: str = "cpu"):
        # device здесь по сути не используется, оставляем для совместимости с кодом
        self.device = device
        self.api_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    def predict_pdb(self, sequence: str) -> str:
        """
        Отправляет последовательность на ESMFold API и возвращает PDB-строку.
        Может бросать requests.exceptions.RequestException, если сеть/сервер недоступны.
        """
        seq = sequence.strip()
        if not seq:
            raise ValueError("Empty sequence passed to ESMFoldWrapper.predict_pdb")

        # В официальном примере используется curl --data "SEQ"
        # requests.post с data=seq даёт аналогичное поведение (form-encoded body)
        resp = requests.post(self.api_url, data=seq)
        resp.raise_for_status()
        pdb_str = resp.text
        if not pdb_str.startswith("ATOM") and "HEADER" not in pdb_str:
            # на всякий случай минимальная проверка; ESMFold обычно возвращает валидный PDB
            raise RuntimeError(f"ESMFold API returned unexpected response:\n{pdb_str[:200]}...")
        return pdb_str

