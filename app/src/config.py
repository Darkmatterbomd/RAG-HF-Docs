# config.py
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderSettings(BaseSettings):
    model_name: str = "BarraHome/vmware-embeddings-large-v1"
    batch_size: int = 256
    convert_to_numpy: bool = True
    normalize_embeddings: bool = True
    show_progress_bar: bool = True


class ChunkingSettings(BaseSettings):
    chunk_size: int = 400
    chunk_overlap: int = 50
    #min_chunk_tokens: int = 20


class IndexSettings(BaseSettings):
    index_path: Path = Path("data/index/index.faiss")
    metadata_path: Path = Path("data/index/metadata.jsonl")

    bm25_dir_path: Path = Path("data/bm25/")
    k1: float = 1.5
    b: float = 0.75


class EvalSettings(BaseSettings):
    eval_path: Path = Path("data/eval/eval_data.jsonl")

class GeneratorSettings(BaseSettings):
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 128
    do_sample: bool = False
    
    system_text: str = (
    """You are an expert in HuggingFace Transformers.
    Use the provided context as the main source of facts.
    If the context is insufficient, you can supplement
    the answer with your own knowledge, but clearly indicate this.

        Priorities:
        1. Accuracy (don’t make up APIs and parameters)
        2. Completeness (examples, use cases, related classes)
        3. Structured format (lists, code, source references)"""
      )

    torch_dtype:str ="bfloat16"


    # quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    device: str = "cuda"

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    generator: GeneratorSettings = Field(default_factory=GeneratorSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)



