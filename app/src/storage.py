import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_metadata(
        chunks: list[str],
        sources: list[str],
) -> list[dict]:
    
    counter = defaultdict(int)
    idx = []
    for s in sources:
        idx.append(f"{s}::{counter[s]}")
        counter[s] += 1

    
    return [
        {"doc_id": index, "chunk":c, "source":s}
    for index, c, s in zip(idx, chunks, sources)
    ]

def save_metadata(
    metadata: list[dict],
    metadata_path: Path,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        for line in metadata:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        logger.exception(f"Metadata file not found: {metadata_path}", exc_info=True)
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        md = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Metadata has been successfully loaded from {str(metadata_path)}")
        return md