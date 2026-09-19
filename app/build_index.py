from sentence_transformers import SentenceTransformer
from functools import partial
from transformers import AutoTokenizer
from torch.cuda import is_available

from src.config import Settings
from src.data_loader import *
from src.utils import *
from src.chunking import *
from src.embedder import *
from src.storage import *
from src.retriever import *

def build_index(s: Settings):

    device = s.device if is_available() else "cpu"
    embedder = SentenceTransformer(s.embedder.model_name, device=device)
    e_tokenizer = AutoTokenizer.from_pretrained(s.embedder.model_name)


    dataset = load_dataset()
    dataset = dataset.map(
        texts_cleaner,
        batched=True,
        remove_columns=dataset.column_names
    )
    chunked_dataset = dataset.map(
        make_chunker(
            chunk_size=s.chunking.chunk_size,
            chunk_overlap=s.chunking.chunk_overlap,
            len_fn=partial(tokenized_length, tokenizer=e_tokenizer)
        ),
        batched=True,
        remove_columns=dataset.column_names
    )
    metadata = get_metadata(chunked_dataset["chunks"], chunked_dataset["sources"])
    save_metadata(metadata, s.index.metadata_path)

    embs = get_embeddings(
        model=embedder,
        chunks=chunked_dataset["chunks"],
        batch_size=s.embedder.batch_size,
        convert_to_np=s.embedder.convert_to_numpy,
        normalize_embeddings=s.embedder.normalize_embeddings,
        show_progress_bar=s.embedder.show_progress_bar
    )
    dense_index = FAISSIndex(embs.shape[1])
    dense_index.add(embs)
    dense_index.save(s.index.index_path)

    bm25_index = BM25Index(chunked_dataset["chunks"], k1=s.index.k1, b=s.index.b)
    bm25_index.save(s.index.bm25_dir_path)


if __name__ == "__main__":
    s = Settings()
    build_index(s)


