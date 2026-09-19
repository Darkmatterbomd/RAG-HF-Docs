from sentence_transformers import SentenceTransformer

def get_embeddings(
        model: SentenceTransformer,
        chunks,
        batch_size,
        convert_to_np: bool=True,
        normalize_embeddings: bool=True,
        show_progress_bar: bool=True

):
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        convert_to_numpy=convert_to_np,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar
    )
    return embeddings