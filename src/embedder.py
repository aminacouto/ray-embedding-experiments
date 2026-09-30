"""Um modelo por actor; uma thread de CPU por modelo."""
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.model = SentenceTransformer(model, device="cpu")
        # Aquecimento de cada modelo, incluído no tempo total da rodada.
        self.model.encode(["Warmup text"], show_progress_bar=False)

    def __call__(self, batch):
        texts = batch["text"].tolist()
        vectors = self.model.encode(
            texts, batch_size=len(texts), convert_to_numpy=True,
            show_progress_bar=False,
        )
        return {"id": batch["id"], "embedding": vectors}
