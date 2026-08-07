from fastapi import FastAPI

from rag_bot import answer_question
from schemas import ChatRequest, ChatResponse


app = FastAPI()


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.post('/api/chat')
def chat(question: ChatRequest) -> ChatResponse:
    return answer_question(question.question)

@app.post('/api/upload')
def upload_file():
    pass