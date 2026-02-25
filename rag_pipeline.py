# rag_pipeline.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore():
    loader = DirectoryLoader("docs/", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ Vector store built!")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local("faiss_index", embeddings,
                            allow_dangerous_deserialization=True)

def retrieve(query, vectorstore, k=3):
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = [r[0].page_content for r in results]
    scores = [r[1] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 999
    confidence = "high" if avg_score < 1.2 else "low"
    return docs, confidence