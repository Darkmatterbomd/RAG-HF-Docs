import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Callable

logger = logging.getLogger(__name__)

def make_chunker(
    chunk_size:int,
    chunk_overlap:int,
    len_fn: Callable   
):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n```\n", "\n```", "```", "\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len_fn,
    )
    def _chunk_batch(
        examples: dict[str, list[str]],
        ) -> dict[str, list[str]]:

        all_chunks, all_sources = [], []
        texts = examples['text']
        for text, source in zip(texts, examples.get("source", [None]*len(examples["text"]))):
            chunks = splitter.split_text(text)
            chunks = [c.strip() for c in chunks if len_fn(c.strip()) > 20]
            if chunks != []:
                all_chunks.extend(chunks)
            sources = [source or 'unknown'] * len(chunks)
            all_sources.extend(sources)
            
        return {
            "chunks": all_chunks,
            "sources": all_sources
        }
    logger.info("Chukning has been completed successefly.")
    return _chunk_batch

