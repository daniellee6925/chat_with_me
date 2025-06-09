# Ask Me Anything AI – Daniel Lee

A personalized AI chatbot built with LangChain that represents Daniel Lee. It uses Retrieval-Augmented Generation (RAG) to answer questions about Daniel’s background, career, projects, and goals — based solely on documents written by Daniel himself.

## Features

- **RAG-based question answering** grounded in Daniel's personal content
- **Conversational memory** using LangChain agents and history-aware retrieval
- **Custom chunked documents** organized by topic (e.g., Personal, Projects, Education)
- **Safe responses only** — avoids hallucination by only answering from trusted sources

## Tech Stack

- LangChain (with LangGraph agents)
- ChromaDBfor vector search
- OpenAI LLMs
- JSON source content

## Demo Idea

Ask things like:
- “Where is Daniel from?”
- “What projects has Daniel built?”
- “Why did he switch from actuarial science to ML?”