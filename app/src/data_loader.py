from pathlib import Path
import glob
from datasets import Dataset
import logging


logger = logging.getLogger(__name__)

def load_dataset(path:str='rag_dataset/data') -> Dataset:
  path = Path(path) 
  try:

    documents = []
    for filepath in path.glob('*.md'):
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append({
                'source': str(filepath),
                'text': f.read()
            })
  
    logger.info(f"Loaded {len(documents)} documents")
    dataset = Dataset.from_list(documents)

    return dataset
  
  except FileNotFoundError:
    logger.exception(f'File {path} dont found. Please go "git clone https://github.com/firebolt-db/rag_dataset.git"', exc_info=True)
    


 