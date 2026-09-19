from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)
class EvalRetriever:

    def __init__(self, eval_data_path:Path):

        try:
            with eval_data_path.open("r", encoding="utf-8") as f:
                eval_data = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Evaluation data has been successfully loaded from {str(eval_data_path)}")
        except Exception as e:
            logger.exception(f"Exception while load evaluation data from {str(eval_data_path)}: {e}.")
            raise e
        self.questions = [ex["question"] for ex in eval_data]
        self.correct_chunk_id = [ex["relevant_chunk_ids"] for ex in eval_data]
        self.n_q = len(self.correct_chunk_id)


    def _check_length(self, retriever_chunk_id):
        if len(retriever_chunk_id) != self.n_q:
            raise ValueError("retriever_chunk_id and correct_chunk_id must have the same length.")
            

    def recall(self, retriever_chunk_id, k:Optional[int]=None):

        self._check_length(retriever_chunk_id)
        recalls = []
        for i in range(self.n_q):

            retrieved = retriever_chunk_id[i]
            if k is not None:
                retrieved = retrieved[:k]

            n_rel = len(set(retrieved) & set(self.correct_chunk_id[i]))
            n_relevant =len(set(self.correct_chunk_id[i]))
            recalls.append(n_rel/n_relevant if n_relevant > 0 else 0.0)

        return sum(recalls)/self.n_q

    def precision(self, retriever_chunk_id, k:Optional[int]=None, top_k:int=5):
        self._check_length(retriever_chunk_id)

        precisions = []
        for i in range(self.n_q):

            retrieved = retriever_chunk_id[i]
            if k is not None:                 
                retrieved = retrieved[:k]
                top_k = k

            n_rel = len(set(retrieved) & set(self.correct_chunk_id[i]))

            precisions.append(n_rel/top_k if top_k > 0 else 0.0)
        return sum(precisions)/self.n_q


    def mrr(self, retriever_chunk_id):

        self._check_length(retriever_chunk_id)

        reciprocal_ranks = []
        for i in range(self.n_q):
            retrieved = retriever_chunk_id[i] 
            relevant = set(self.correct_chunk_id[i])
            rr = 0.0
            for rank, chunk_id in enumerate(retrieved, start=1):
                if chunk_id in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return sum(reciprocal_ranks) / self.n_q

    def hit_rate(self, retriever_chunk_id, k:Optional[int]=None, top_k: int = 5):
        self._check_length(retriever_chunk_id)
        hits = []
        for i in range(self.n_q):
            retrieved = set(retriever_chunk_id[i][:top_k])
            if k is not None:
                retrieved = set(retriever_chunk_id[i][:k])
            relevant = set(self.correct_chunk_id[i])
            hits.append(1.0 if retrieved & relevant else 0.0)
        return sum(hits) / self.n_q

    def get_questions(self):
        return self.questions