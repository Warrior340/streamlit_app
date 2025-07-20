import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.qparser import syntax
from whoosh.analysis import StemmingAnalyzer, StopFilter
import os
import pickle
import hashlib
import json
from transformers import AutoTokenizer
from docling_core.transforms.chunker import HierarchicalChunker
from docling.document_converter import DocumentConverter
from pymilvus import MilvusClient
import fitz
import time
from typing import List, Tuple, Dict
from whoosh.scoring import BM25F
from tqdm import tqdm
from pathlib import Path

# Download NLTK data if not already present

client = ollama.Client(host='http://10.144.177.192:12345')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Define paths to local models
local_model_path = "./models_for_transfer/"
# Define constants
CHUNK_SIZE = 512
OVERLAP = 128
TOKEN_LIMIT = 8000
MAX_DOCUMENTS = 5  # Maximum number of documents allowed
EMBEDDER_MODEL = f"{local_model_path}/all-MiniLM-L6-v2"
RERANKER_MODEL = f"{local_model_path}/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "glm4:latest"

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Persistent storage paths
STORAGE_DIR = "./persistent_storage"
DOCUMENT_STORE_FILE = os.path.join(STORAGE_DIR, "document_store.json")

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

# Cache model loading
@st.cache_resource
def load_models():
    """Load and cache embedding and reranker models"""
    with st.spinner("Loading embedding model..."):
        embedder = SentenceTransformer(EMBEDDER_MODEL, trust_remote_code=True, local_files_only=True)
    
    with st.spinner("Loading reranker model..."):
        reranker = CrossEncoder(RERANKER_MODEL, trust_remote_code=True, local_files_only=True)
    
    return embedder, reranker

def emb_text(text):
    return embedder.encode(text).tolist()

def get_document_id(filename: str, file_content: str) -> str:
    """Generate a unique document ID based on filename and content hash"""
    content_hash = hashlib.md5(file_content.encode()).hexdigest()
    return hashlib.md5(f"{filename}_{content_hash}".encode()).hexdigest()[:16]

def get_collection_name(doc_id: str) -> str:
    """Generate collection name for a document"""
    return f"doc_{doc_id}"

def get_index_dir(doc_id: str) -> str:
    """Generate index directory for a document"""
    return f"indexdir_{doc_id}"

def save_document_store():
    """Save document store to persistent storage"""
    try:
        with open(DOCUMENT_STORE_FILE, 'w') as f:
            json.dump(st.session_state.document_store, f, indent=2)
    except Exception as e:
        st.error(f"Error saving document store: {e}")

def load_document_store():
    """Load document store from persistent storage"""
    if os.path.exists(DOCUMENT_STORE_FILE):
        try:
            with open(DOCUMENT_STORE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading document store: {e}")
            return {}
    return {}

def check_admin_login():
    """Check if user is logged in as admin"""
    return st.session_state.get('is_admin', False)

def admin_sidebar():
    """Admin login interface in sidebar"""
    st.sidebar.title("Admin Panel")
    
    if not check_admin_login():
        st.sidebar.subheader("🔐 Admin Login")
        
        with st.sidebar.form("admin_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.sidebar.success("Admin login successful!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")
    else:
        st.sidebar.success("✅ Logged in as Admin")
        if st.sidebar.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

def admin_document_management():
    """Admin document management interface"""
    st.subheader("📁 Document Management")
    
    # Show document limit status
    doc_count = len(st.session_state.document_store)
    st.write(f"**Documents: {doc_count}/{MAX_DOCUMENTS}**")
    
    # Progress bar for document limit
    progress_value = doc_count / MAX_DOCUMENTS
    st.progress(progress_value)
    
    if doc_count >= MAX_DOCUMENTS:
        st.warning(f"⚠️ Maximum document limit ({MAX_DOCUMENTS}) reached. Remove some documents to add new ones.")
    
    # Show existing documents
    if st.session_state.document_store:
        st.write("**Indexed Documents:**")
        for doc_id, doc_info in st.session_state.document_store.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {doc_info['filename']}")
            with col2:
                st.write(f"{doc_info['chunk_count']} chunks")
            with col3:
                if st.button(f"🗑️ Remove", key=f"remove_{doc_id}"):
                    # Remove document
                    try:
                        collection_name = doc_info['collection_name']
                        if st.session_state.milvus_client.has_collection(collection_name):
                            st.session_state.milvus_client.drop_collection(collection_name)
                        
                        # Remove BM25 index directory
                        import shutil
                        index_dir = get_index_dir(doc_id)
                        if os.path.exists(index_dir):
                            shutil.rmtree(index_dir)
                        
                        del st.session_state.document_store[doc_id]
                        save_document_store()
                        st.success(f"Removed document: {doc_info['filename']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing document: {e}")
    else:
        st.info("No documents indexed yet. You can add up to 5 documents.")
    
    # File uploader section
    st.subheader("📤 Add New Documents")
    
    # Only show file uploader if we haven't reached the limit
    if doc_count < MAX_DOCUMENTS:
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=["txt", "md", "rmd"], 
            accept_multiple_files=True,
            help=f"You can upload up to {MAX_DOCUMENTS - doc_count} more document(s)"
        )
        
        if uploaded_files:
            valid_files = []
            for uploaded_file in uploaded_files:
                try:
                    file_extension = uploaded_file.name.split(".")[-1]
                    if file_extension in ["txt", "md", "rmd"]:
                        file_content = uploaded_file.getvalue().decode("utf-8")
                        
                        # Check if document already exists
                        if not check_duplicate_document(uploaded_file.name, file_content):
                            valid_files.append((uploaded_file, file_content))
                        else:
                            st.warning(f"Document '{uploaded_file.name}' with this content is already indexed!")
                    else:
                        st.error(f"Unsupported file format for {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {e}")
            
            if valid_files and st.button("🚀 Index Selected Documents"):
                for uploaded_file, file_content in valid_files:
                    # Save uploaded file
                    file_path = uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    status = index_document(file_content, file_path, uploaded_file.name, file_content)
                    if "successfully" in status:
                        st.success(status)
                    else:
                        st.error(status)
                st.rerun()
    else:
        st.info(f"Maximum document limit ({MAX_DOCUMENTS}) reached. Please remove some documents to add new ones.")

# BM25 Indexing Setup
def setup_bm25(doc_id: str):
    """Setup Whoosh BM25 index for a specific document"""
    schema = Schema(
        content=TEXT(analyzer=StemmingAnalyzer() | StopFilter(), stored=True),
        doc_id=ID(stored=True)
    )
    index_dir = get_index_dir(doc_id)
    
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
        ix = create_in(index_dir, schema)
    else:
        from whoosh import index
        ix = index.open_dir(index_dir)
    
    return ix

def index_bm25(ix, text: str, doc_id: str):
    """Add text to BM25 index with document ID"""
    try:
        writer = ix.writer()
        writer.add_document(content=text, doc_id=doc_id)
        writer.commit()
        return True
    except Exception as e:
        st.error(f"Error indexing to BM25: {e}")
        return False

def create_chunks(doc):
    """Create chunks from document"""
    chunker = HierarchicalChunker()
    
    try:
        texts = []
        c = ""
        for chunk in chunker.chunk(doc):
            for headings in chunk.meta.headings:
                c = c + f"{headings}\n"
            texts.append(f"Heading:{c}\n{chunk.text}")
            c = ""
        return texts
    except Exception as e:
        print(f"Unable to chunk {e}")
        return []

def check_duplicate_document(filename: str, file_content: str) -> bool:
    """Check if document already exists based on content"""
    doc_id = get_document_id(filename, file_content)
    return doc_id in st.session_state.document_store

# Process document with progress indicators
def index_document(document: str, file_path: str, filename: str, file_content: str) -> str:
    """Index document with progress indicators"""
    # Check if we've reached the maximum number of documents
    if len(st.session_state.document_store) >= MAX_DOCUMENTS:
        return f"Maximum number of documents ({MAX_DOCUMENTS}) reached. Please remove some documents before adding new ones."
    
    doc_id = get_document_id(filename, file_content)
    
    # Check if document already exists
    if doc_id in st.session_state.document_store:
        return f"Document '{filename}' with this content is already indexed."
    
    collection_name = get_collection_name(doc_id)
    
    # Setup BM25 index for this document
    ix = setup_bm25(doc_id)
    
    converter = DocumentConverter()
    doc = converter.convert(file_path).document
    chunks = create_chunks(doc)
    
    if not chunks:
        return "No chunks created from document. Please check the content."
    
    embedding_dim = embedder.get_sentence_embedding_dimension()
    
    # Drop existing collection if it exists
    if st.session_state.milvus_client.has_collection(collection_name):
        st.session_state.milvus_client.drop_collection(collection_name)
    
    # Create new collection
    st.session_state.milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",
        consistency_level="Strong",
    )
    
    # Store document metadata
    st.session_state.document_store[doc_id] = {
        'filename': filename,
        'chunks': chunks,
        'collection_name': collection_name,
        'chunk_count': len(chunks),
        'file_content_hash': hashlib.md5(file_content.encode()).hexdigest()
    }
    
    # Process chunks with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"Processing chunk {i + 1}/{len(chunks)}")
        
        embedding = emb_text(chunk)
        data.append({"id": i, "vector": embedding, "text": chunk})
        
        try:
            index_bm25(ix, chunk, doc_id)
        except Exception as e:
            st.error(f"BM25 error: {e}")
    
    # Insert embeddings to Milvus
    st.session_state.milvus_client.insert(collection_name=collection_name, data=data)
    
    if data:
        try:
            progress_bar.progress(1.0)
            status_text.text("Indexing complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            # Save document store to persistent storage
            save_document_store()
            
            return f"Document '{filename}' indexed successfully! Created {len(chunks)} chunks."
        except Exception as e:
            return f"Error adding embeddings to index: {e}"
    else:
        return "No embeddings were created. Indexing failed."

def search_document(query: str, doc_id: str, top_k: int = 5) -> List[Tuple[str, int]]:
    """Search within a specific document, returning chunks with their indices"""
    if doc_id not in st.session_state.document_store:
        return []
    
    doc_info = st.session_state.document_store[doc_id]
    collection_name = doc_info['collection_name']
    
    # Setup BM25 index for this document
    ix = setup_bm25(doc_id)
    
    results = []
    
    # BM25 retrieval
    bm25_results = []
    try:
        with ix.searcher(weighting=BM25F()) as searcher:
            query_parser = QueryParser("content", ix.schema, group=syntax.OrGroup)
            q = query_parser.parse(query)
            bm25_hits = searcher.search(q, limit=30)
            bm25_results = [hit["content"] for hit in bm25_hits]
    except Exception as e:
        st.warning(f"BM25 search error for document {doc_id}: {e}")
    
    # Milvus retrieval
    milvus_results = []
    try:
        query_embedding = emb_text(query)
        search_res = st.session_state.milvus_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=30,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
        milvus_results = [(res["entity"]["text"], res["id"]) for res in search_res[0]]
    except Exception as e:
        st.warning(f"Vector search error for document {doc_id}: {e}")
    
    # Merge results (prioritize Milvus results with chunk indices)
    all_results = []
    for text, chunk_id in milvus_results:
        all_results.append((text, chunk_id))
    
    # Add BM25 results that aren't already included
    for text in bm25_results:
        if not any(text == result[0] for result in all_results):
            # Find chunk index in original chunks
            chunk_idx = -1
            for i, chunk in enumerate(doc_info['chunks']):
                if chunk == text:
                    chunk_idx = i
                    break
            all_results.append((text, chunk_idx))
    
    if not all_results:
        return []
    
    # Rerank
    try:
        texts = [result[0] for result in all_results]
        rerank_pairs = [(query, text) for text in texts]
        scores = reranker.predict(rerank_pairs)
        sorted_results = [all_results[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
        return sorted_results[:top_k]
    except Exception as e:
        st.warning(f"Reranking error for document {doc_id}: {e}")
        return all_results[:top_k]

# Hybrid Search across multiple documents
def hybrid_search_multi_doc(query: str, selected_doc_ids: List[str], top_k: int = 5) -> List[Tuple[str, str, str, int]]:
    """Perform hybrid search across multiple selected documents"""
    all_results = []
    
    for doc_id in selected_doc_ids:
        doc_results = search_document(query, doc_id, top_k)
        doc_info = st.session_state.document_store[doc_id]
        filename = doc_info['filename']
        
        # Add document context to results
        for result_text, chunk_idx in doc_results:
            all_results.append((result_text, filename, doc_id, chunk_idx))
    
    # If we have results from multiple documents, we might want to rerank across all
    if len(all_results) > top_k:
        try:
            texts = [result[0] for result in all_results]
            rerank_pairs = [(query, text) for text in texts]
            scores = reranker.predict(rerank_pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            all_results = [all_results[i] for i in sorted_indices[:top_k]]
        except Exception as e:
            st.warning(f"Cross-document reranking error: {e}")
            all_results = all_results[:top_k]
    
    return all_results

def generate_follow_up_questions(query: str, response: str, context: str) -> List[str]:
    """Generate follow-up questions based on the query, response, and context"""
    try:
        prompt = f"""Based on the following query, response, and context, generate 3 relevant follow-up questions that a user might ask:

Query: {query}
Response: {response}
Context: {context[:1000]}...

Generate 3 concise follow-up questions (each question should be one line):"""
        
        follow_up_response = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"num_ctx": 8000, "temperature": 0.3}
        )
        
        # Parse the response to extract questions
        questions = []
        lines = follow_up_response['message']['content'].strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering and formatting
                question = line.lstrip('123456789. -•')
                if question and '?' in question:
                    questions.append(question)
        
        return questions[:3]
    except Exception as e:
        st.warning(f"Error generating follow-up questions: {e}")
        return []

def format_chat_history_for_context(chat_history: List[Dict], max_turns: int = 3) -> str:
    """Format recent chat history for context"""
    if not chat_history:
        return ""
    
    context_parts = []
    recent_history = chat_history[-max_turns*2:]  # Get last few turns (user + assistant pairs)
    
    for entry in recent_history:
        role = entry['role']
        content = entry['content']
        if role == 'user':
            context_parts.append(f"User: {content}")
        else:
            context_parts.append(f"Assistant: {content}")
    
    return "\n".join(context_parts)

# RAG Query Processing
def retrieve_and_generate_multi_doc(query: str, selected_doc_ids: List[str], 
                                   top_k: int = 5, token_limit: int = TOKEN_LIMIT) -> Tuple[str, List[Tuple[str, str, str, int]]]:
    """Retrieve relevant chunks from multiple documents and generate response"""
    if not selected_doc_ids:
        return "No documents selected. Please select at least one document to search.", []
    
    # Get relevant chunks from selected documents
    with st.spinner("Retrieving relevant information from selected documents..."):
   
