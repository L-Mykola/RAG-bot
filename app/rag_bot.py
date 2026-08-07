import os
from dotenv import load_dotenv
from typing import List

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from schemas import ChatRequest, ChatResponse, AnswerWithSources

load_dotenv()


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.environ.get("OPENAI_API_KEY"))


structured_llm = llm.with_structured_output(AnswerWithSources)

SYSTEM_PROMPT = """You are an assistant for question-answering tasks.
Use only the numbered documentation excerpts below to answer the question.
If the answer isn't in the context, say you don't know — don't make anything up, and return an empty used_excerpt_ids list.
Answer in the same language the question was asked in. Keep the answer concise (max 3-4 sentences).
In used_excerpt_ids, list only the excerpt numbers you actually drew on to construct the answer — not every excerpt you were given.

Context:
{context}"""


def format_source(metadata: dict) -> str:
    section = metadata.get("section") or "Document"
    subsection = metadata.get("subsection")
    page = metadata.get("page")
    label = f'Section "{section}"'
    if subsection:
        label += f', Subsection "{subsection}"'
    if page:
        label += f", Page {int(page)}"
    return label


def answer_question(question: str) -> ChatResponse:
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.55},
    )
    docs = retriever.invoke(question)

    if not docs:
        answer = "I couldn't find an answer to this question in the documentation."
        sources = []
    else:
        numbered_context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
        system_prompt = SYSTEM_PROMPT.format(context=numbered_context)

        result: AnswerWithSources = structured_llm.invoke([SystemMessage(system_prompt), HumanMessage(question)])
        answer = result.answer

        used_docs = [docs[i - 1] for i in result.used_excerpt_ids if 1 <= i <= len(docs)]
        if not used_docs:
            used_docs = docs

        seen = set()
        sources = []
        for d in used_docs:
            label = format_source(d.metadata)
            if label not in seen:
                seen.add(label)
                sources.append(label)


    return ChatResponse(answer=answer, sources=sources)

