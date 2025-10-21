# 📘 StudyBuddy — RAG PDF Q&A Chatbot.

Retrieval-Augmented Generation (RAG) chatbot for PDF Q&amp;A with a working FastAPI backend, FAISS indexer, LLM reasoning layer, and a Streamlit GUI. 

StudyBuddy is an intelligent PDF Q&A assistant that combines **semantic search** with **LLM reasoning**.
Upload your lecture notes or readings and ask natural-language questions, StudyBuddy retrieves the most
relevant chunks and generates **cited, grounded answers**.

## 🚀 Features
- 🔍 **Retrieval-Augmented Generation (RAG):** FAISS-based vector search over chunked PDF text.
- 🧠 **LLM Integration:** Uses FLAN-T5-XL for context aware answers.
- 📚 **Citation Support:** Every response includes exact PDF references for transparency.
- 🖥️ **Streamlit Front-End:** Simple, clean, asthetically pleasing drag-and-drop UI for upload, ingest, and chat.
- ⚙️ **FastAPI Backend:** Modular endpoints for ingesting, searching, and answering.

## 🏗️ Project Structure
- Main.py # Bootstrap / environment setup.
- api.py # FastAPI endpoints.
- GUI.py # Streamlit chat interface.
- pdfToChunks.py # PDF parsing + semantic chunking.
- VectorDB.py # FAISS index creation + retrieval.
- LLM.py # Model loading + response generation.
- trainingBuilding.py # Fine-tuning / training scripts.

## 📦 Tech Stack

Python • FIASS • Json • Pickle • numpy • Hugging Face Transformers • Pytorch • OS
Path • Math • pypdf • pytesseract • pdf2image • subprocess • shutil • re • time 
Hugging Face Sentence Transformers • streamlit • requests • fastapi • pydantic 


## 🧩 Example Query
Q: “What are the main goals of an operating system?” 
A: Convenience for users and efficient system operation. Citations: Operating_Systems_Week1.pdf, Slide 54, Operating_Systems_Week1.pdf, Slide 55, Operating_Systems_Week1.pdf, Slide 56

## 📈 Model Evaluation

| Metric | Value |
|--------|--------|
| Eval Loss | **1.182** |
| Perplexity | **3.26** |
| Eval Runtime | 11.5 s |
| Samples / sec | 34.0 |
| Steps / sec | 4.26 |

The fine-tuned FLAN-T5 model achieved a **perplexity of 3.26**, showing strong comprehension of uploaded PDF text.
This demonstrates that the RAG pipeline (FAISS + LLM) effectively grounds answers in the retrieved context while maintaining fluent, accurate generation.
