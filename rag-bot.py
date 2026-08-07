import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("NiftyBridge Docs Chatbot")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.environ.get("OPENAI_API_KEY"))


class AnswerWithSources(BaseModel):
    answer: str = Field(description="Answer to the user's question, in the same language the question was asked in.")
    used_excerpt_ids: List[int] = Field(
        description="Numbers of the context excerpts (e.g. [1], [2]) actually used to build the answer. "
        "Empty list if the answer isn't in the context."
    )


structured_llm = llm.with_structured_output(AnswerWithSources)

SYSTEM_PROMPT = """You are an assistant for question-answering tasks.
Use only the numbered documentation excerpts below to answer the question.
If the answer isn't in the context, say you don't know — don't make anything up, and return an empty used_excerpt_ids list.
Answer in the same language the question was asked in. Keep the answer concise (max 3-4 sentences).
In used_excerpt_ids, list only the excerpt numbers you actually drew on to construct the answer — not every excerpt you were given.

Context:
{context}"""


def format_source(metadata: dict) -> str:
    section = metadata.get("section") or "Документ"
    subsection = metadata.get("subsection")
    page = metadata.get("page")
    label = f'Розділ "{section}"'
    if subsection:
        label += f', підрозділ "{subsection}"'
    if page:
        label += f", сторінка {int(page)}"
    return label

    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
            sources = getattr(message, "sources", None)
            if sources:
                st.markdown("**Джерела:**\n" + "\n".join(f"- {s}" for s in sources))

prompt = st.chat_input("Постав запитання по документації...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.55},
    )
    docs = retriever.invoke(prompt)

    if not docs:
        answer = "Не знайшов відповіді на це питання в документації."
        sources = []
    else:
        numbered_context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
        system_prompt = SYSTEM_PROMPT.format(context=numbered_context)

        result: AnswerWithSources = structured_llm.invoke([SystemMessage(system_prompt), HumanMessage(prompt)])
        answer = result.answer

        used_docs = [docs[i - 1] for i in result.used_excerpt_ids if 1 <= i <= len(docs)]
        # fallback: model didn't cite anything usable — better to show all
        # retrieved sources than none, so the user still sees where to look
        if not used_docs:
            used_docs = docs

        seen = set()
        sources = []
        for d in used_docs:
            label = format_source(d.metadata)
            if label not in seen:
                seen.add(label)
                sources.append(label)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.markdown("**Джерела:**\n" + "\n".join(f"- {s}" for s in sources))

    ai_message = AIMessage(answer)
    ai_message.sources = sources
    st.session_state.messages.append(ai_message)
