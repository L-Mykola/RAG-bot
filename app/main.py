from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from loguru import logger

from rag_bot import answer_question
from schemas import ChatRequest, ChatResponse
from ingestion import ingest_file, ingest_all, DOCUMENTS_DIR
from loaders import LOADERS


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR.parent / "logs"
logger.add(LOGS_DIR / "app.log", rotation="10 MB", retention="7 days")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest_all()
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"title": "NiftyBridge Docs Chatbot"}
    )


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.post('/api/chat')
def chat(question: ChatRequest) -> ChatResponse:
    logger.info(f"chat request: question={question.question!r}")
    try:
        response = answer_question(question.question)
    except Exception:
        logger.exception("chat request failed")
        raise HTTPException(status_code=500, detail="Internal server error")
    logger.info(f"chat response: answer={response.answer!r} sources={response.sources!r}")
    return response


@app.post('/api/upload')
async def upload_file(file: UploadFile = File(...)):
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in LOADERS:
        supported = ", ".join(sorted(LOADERS))
        logger.warning(f"upload rejected: unsupported extension {ext!r}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Supported: {supported}")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DOCUMENTS_DIR / safe_name

    try:
        with dest_path.open("wb") as f:
            f.write(await file.read())
        chunk_count = ingest_file(str(dest_path))
    except Exception:
        logger.exception(f"upload failed for file={safe_name!r}")
        raise HTTPException(status_code=500, detail="Internal server error")

    logger.info(f"uploaded file={safe_name!r} chunks_added={chunk_count}")
    return {"filename": safe_name, "chunks_added": chunk_count}
