from functools import lru_cache
from src.config import Settings
from src.pipeline import QueryPipeline


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_pipeline() -> QueryPipeline:
    return QueryPipeline(get_settings())