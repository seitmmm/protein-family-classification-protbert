from transformers import BertModel, BertTokenizer
import torch
import numpy as np


class ProtBertEmbedder:
    """
    Wrapper around Rostlab/prot_bert to get CLS embeddings
    """

    def __init__(self, model_name: str = "Rostlab/prot_bert",
                 device: str | None = None,
                 max_length: int = 1024):
        # auto-select device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=False
        )
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # IMPORTANT: disable dropout, etc.

    def _clean_sequence(self, sequence: str) -> str:
        # remove spaces/newlines and uppercase
        seq = sequence.replace(" ", "").replace("\n", "").upper()
        # replace non-standard amino acids with X
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        seq = "".join(ch if ch in valid_aa else "X" for ch in seq)
        return seq

    def get_embedding(self, sequence: str) -> np.ndarray:
        """
        Returns CLS embedding as 1D numpy array
        """
        seq = self._clean_sequence(sequence)
        seq_spaced = " ".join(list(seq))

        inputs = self.tokenizer(
            seq_spaced,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()