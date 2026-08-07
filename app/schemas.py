from pydantic import BaseModel, Field

class AnswerWithSources(BaseModel):
    answer: str = Field(description="Answer to the user's question, in the same language the question was asked in.")
    used_excerpt_ids: list[int] = Field(
        description="Numbers of the context excerpts (e.g. [1], [2]) actually used to build the answer. "
        "Empty list if the answer isn't in the context."
    )

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


