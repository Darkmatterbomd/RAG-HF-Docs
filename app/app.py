import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.dependencies import get_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading pipeline...")
    get_pipeline()  
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RAG HF Docs API",
    description="Q&A по документации HuggingFace Transformers с RAG + квантизованной LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")




@app.get("/")
def root():
    return {"message": "RAG API is running. See /docs"}