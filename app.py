import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
import json
from dotenv import load_dotenv

load_dotenv()


def get_vectorstore_from_file(file):
    # get the text in document form
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to LangChain Document format
    docs = [Document(page_content=item["text"]) for item in data]

    # create a vector store from the chunks
    vector_store = Chroma.from_documents(
        docs, OpenAIEmbeddings(), collection_metadata={"hnsw:space": "cosine"}
    )

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    # retriever will be used to perform a semantic search on the vector DB
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # creating a prompt template for history-aware retrieval
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are an AI assistant that helps answer questions about Daniel Lee "
                    "by generating highly relevant search queries based on the conversation. "
                    "Daniel Lee is an actuary, machine learning engineer, and builder of AI tools. "
                    "Use the conversation so far to infer what information is being requested "
                    "about Daniel and generate a concise, context-aware search query to retrieve it."
                )
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "system",
                "Generate a search query to retrieve the most relevant information about Daniel Lee based on the above conversation.",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI assistant that answers questions about Daniel Lee. "
                    "Only use the information provided in the context below. "
                    "If the answer is not found in the context, say 'Sorry, I don't have an answer. Daniel hasn't provided that information yet.'\n\n"
                    "Context:\n{context}"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_query}
    )
    return response["answer"]


# ---------------------------------------------------------------------------

# app config
st.set_page_config(page_title="Chat with Daniel Lee")
st.title("Chat with Daniel Lee")


# session state
if "chat_history" not in st.session_state:
    # state does not change whenever you re-read application
    st.session_state.chat_history = [
        AIMessage(
            content="Hi there! I'm Daniel Lee — ask me anything about my background, projects, or goals."
        )
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_file("chunked_output.json")

    # user input
user_query = st.chat_input("Type your message here...")  # creates chat input
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
