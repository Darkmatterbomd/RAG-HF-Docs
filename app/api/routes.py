from fastapi import APIRouter, Depends, HTTPException

from api.schemas import AskRequest, AskResponse, SourceChunk, HealthResponse, REvalResponse, REvalRequest
from api.dependencies import get_pipeline, get_settings
from src.pipeline import QueryPipeline


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(pipeline: QueryPipeline = Depends(get_pipeline),
           settings=Depends(get_settings)):
    return HealthResponse(
        status="ok",
        index_size=pipeline.index.index.ntotal,
        device=pipeline.device,
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, pipeline: QueryPipeline = Depends(get_pipeline)):
    try:
        result = pipeline.ask(payload.question, top_k=payload.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        question=payload.question,
        answer=result["answer"],
        sources=[SourceChunk(**c) for c in result["sources"]],
    )


@router.post("/retrieve", response_model=list[SourceChunk])
def retrieve(payload: AskRequest, pipeline: QueryPipeline = Depends(get_pipeline)):
    chunks = pipeline.retrieve(payload.question, top_k=payload.top_k)
    return [SourceChunk(**c) for c in chunks]


@router.post("/eval_retiever", response_model=REvalResponse)
def r_eval(payload:REvalRequest, pipeline: QueryPipeline = Depends(get_pipeline)):
    result = pipeline.eval_retriever(payload.top_k, payload.k)
    return REvalResponse(
        recall=result["recall"],
        precision=result["precision"],
        hit_rate=result["hit_rate"],
        mrr=result["MRR"]
    )