from transformers import AutoTokenizer
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def tokenized_length(text: str, tokenizer: AutoTokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def texts_cleaner(samples) -> list[str]:

  https_template = r'https?://[^"\s]+'
  img_template = r'<img\s+src="' + https_template + r'"\s+alt="([^"]*)"\s*/>\n\n'
  clear_texts = []
  for t in samples["text"]:
    clear_str = re.sub(img_template, "", str(t).strip())
    clear_str = re.sub(r"[/(]*" + https_template, "", clear_str)
    clear_texts.append(clear_str)
  return {
    "text": clear_texts,
    "source": samples["source"]
  }

