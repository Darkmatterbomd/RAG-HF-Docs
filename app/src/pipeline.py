from src.config import Settings
from src.storage import load_metadata
from src.retriever import *
from src.generator import *
from src.eval import *


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from typing import Optional
from pathlib import Path

class QueryPipeline:
    def __init__(self, s:Settings):
        self.s = s

        if not s.index.metadata_path.exists(): 
            raise FileNotFoundError
        self.md = load_metadata(s.index.metadata_path)

        if not s.index.index_path.exists(): 
            raise FileNotFoundError    
        self.index = FAISSIndex.load(s.index.index_path)

        if not s.index.bm25_dir_path.exists():
            raise FileNotFoundError
        self.bm25 = BM25Index.load(s.index.bm25_dir_path)

        if not s.eval.eval_path.exists():
            raise FileNotFoundError
        self.evaluator = EvalRetriever(s.eval.eval_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(s.embedder.model_name, device=self.device)

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=s.generator.load_in_4bit,
                bnb_4bit_use_double_quant=s.generator.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=getattr(torch, s.generator.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=s.generator.bnb_4bit_quant_type
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                s.generator.model_name, 
                quantization_config=bnb_config, 
                device_map="auto"
                )

        else: 
            self.generator = AutoModelForCausalLM.from_pretrained(
                s.generator.model_name, 
                device_map="auto",
                torch_dtype=getattr(torch, s.generator.torch_dtype),
                low_cpu_mem_usage=True
                ).eval()

        self.g_tokenizer = AutoTokenizer.from_pretrained(s.generator.model_name)

    def retrieve(self, question:str, top_k:int=5):
        q_emb = self.embedder.encode(
            [question],
            normalize_embeddings=self.s.embedder.normalize_embeddings
            )
        dense_result = self.index.search(q_emb, top_k) 
        bm25_result = self.bm25.retrieve(question, top_k=top_k)

        hybrid_search_result = hybrid_search_rel_docs(dense_result, bm25_result, top_k=top_k)
        return [
            {
                "doc_id": self.md[int(i)]["doc_id"],
                "source": self.md[int(i)]["source"],
                "chunk": self.md[int(i)]["chunk"],
                "score": float(s)
            }
                for i, s in hybrid_search_result
            ]

    def ask(self, question:str, top_k:int=5):

        context = self.retrieve(question, top_k)
        
        messages = build_prompt(question, system_text=self.s.generator.system_text, chunks=context)

        device = torch.device(self.device)
        answer = run_rag(
            messages=messages,
            generator=self.generator,
            g_tokenizer=self.g_tokenizer,
            device=device,
            max_new_tokens=self.s.generator.max_new_tokens,
            do_sample=self.s.generator.do_sample
            )
        return {"answer": answer, "sources": context} 

    def eval_retriever(self, top_k:int = 5, k:Optional[int]=None):
        questions = self.evaluator.get_questions()
        retriever_chunk_id = [[self.retrieve(q, top_k)[i]["doc_id"] for i in range(top_k)] for q in questions]

        recall = self.evaluator.recall(retriever_chunk_id, k=k)
        precision = self.evaluator.precision(retriever_chunk_id, k=k, top_k=k)
        hit = self.evaluator.hit_rate(retriever_chunk_id, k=k, top_k=top_k)
        mrr = self.evaluator.mrr(retriever_chunk_id)
        return {"recall": recall, "precision": precision, "hit_rate": hit, "MRR": mrr}



