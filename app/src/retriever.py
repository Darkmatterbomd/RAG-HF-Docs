import faiss
import bm25s
import logging
import numpy as np
from rankops import rrf
from typing import Optional
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

class FAISSIndex:
  def __init__(self, dim:int, d_metric:Literal['L2', 'IP']='IP'):

      if d_metric == 'L2':
         self.index = faiss.IndexFlatL2(dim)
      elif d_metric == 'IP':
         self.index = faiss.IndexFlatIP(dim)

      else:
         logger.exception(f"Not valid IndexFlat distance metric: {d_metric}. Should be 'L2'or 'IP'.", exc_info=True)
         raise TypeError("Not valid parameter.")

      logger.info(f"FlatIndex{d_metric} has been successfully created")



        
  def add(self, embs: np.ndarray):
    embs = np.ascontiguousarray(embs, dtype='float32')
    self.index.add(embs)
    logger.info(f"Embeddings has been successefly added to index.")
    return 

  def search(self, q_emb: np.ndarray, top_k:int=5) -> list[tuple]:
     q_emb = np.ascontiguousarray(q_emb, dtype='float32')
     scores, ids = self.index.search(q_emb, top_k)
     return list(zip(map(str, ids[0]), scores[0]))


  def save(self, index_path:Path):
     index_path.parent.mkdir(parents=True, exist_ok=True)
     faiss.write_index(self.index, str(index_path))
     logger.info(f"Index has been successeflly saved to {str(index_path)}")
     return 

  @classmethod
  def load(cls, index_path: Path, d_metric:Literal['L2', 'IP']='IP'):
   if not index_path.exists():
      raise FileNotFoundError(f"Index file not found: {index_path}")
        
   obj = cls.__new__(cls)
   obj.index = faiss.read_index(str(index_path))
   obj.d_metric = d_metric
   return obj


class BM25Index:

   def __init__(self, chunks:list[str], k1:float=1.5, b:float=0.75):
      self.retriever = bm25s.BM25(k1=k1, b=b)   
      tok_chunks = bm25s.tokenize(chunk for chunk in chunks)
      self.retriever.index(tok_chunks)


      self.k1 = k1
      self.b = b 

   def save(self, bm25_dir_path: Path):
      bm25_dir_path.parent.mkdir(parents=True, exist_ok=True)
      self.retriever.save(save_dir=bm25_dir_path)
      logger.info(f"BM25Index has been successeflly saved to dir {str(bm25_dir_path)}")
      return

   @classmethod
   def load(cls, bm25_path: Path):
      if not bm25_path.exists():
         raise FileNotFoundError(f"Index file not found: {bm25_path}")
      obj = cls.__new__(cls)
      obj.retriever = bm25s.BM25().load(save_dir=bm25_path)
      return obj

   def retrieve(self, q:str, top_k:int=5)-> list[tuple]:
      q_tokens = bm25s.tokenize([q])
      ids, scores = self.retriever.retrieve(q_tokens, k=top_k) #coprus is not provided so it must return ids
      return list(zip(map(str, ids[0]), scores[0])) #for rrf



def hybrid_search_rel_docs(dense:list[tuple], bm25:list[tuple], top_k:int=5)->list[tuple]:
   try:
      return rrf(dense, bm25, top_k=top_k)
   except Exception as e:
      logger.exception("Exception while do RRF: {e}", exc_info=True)






      
      
