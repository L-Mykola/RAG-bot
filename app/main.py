from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from rag_bot import answer_question
from schemas import ChatRequest, ChatResponse


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
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
    return answer_question(question.question)

@app.post('/api/upload')
def upload_file():
    pass