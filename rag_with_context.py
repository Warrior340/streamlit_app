import streamlit as st
import nltk
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import ollama
import torch
import re
from typing import List, Tuple
from sklearn.preprocessing import normalize
from functools import lru_cache

# Configuration
MAX_DOC_SIZE = 1000000  # 1MB
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-electra-base"
TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
CHUNK_SIMILARITY_THRESHOLD = 0.82
NUM_RERANK = 50

# Initialize models
@st.cache_resource
def load_models():
    nltk.download('punkt')
    embedder = AutoModel.from_pretrained(EMBEDDING_MODEL)
    cross_model = CrossEncoder(CROSS_ENCODER)
    return embedder, cross_model

embedder, cross_model = load_models()

# Utility functions
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@st.cache_data(max_entries=5)
def get_embeddings(texts: List[str]) -> np.ndarray:
    inputs = TOKENIZER(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = embedder(**inputs)
    return normalize(mean_pooling(outputs, inputs['attention_mask']).numpy(), norm='l2')

# Core processing
def process_document(text: str) -> Tuple[str, List[str]]:
    """Full document processing pipeline"""
    text = clean_text(text)
    chunks = semantic_chunking(text)
    doc_summary = generate_document_summary(chunks)
    return doc_summary, create_contextual_chunks(chunks, doc_summary)

def clean_text(text: str) -> str:
    """Clean and normalize input text"""
    text = re.sub(r'\s+', ' ', text)
    return text[:MAX_DOC_SIZE]

def semantic_chunking(text: str) -> List[str]:
    """Document-aware semantic chunking"""
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [text]
    
    embeddings = get_embeddings(sentences)
    chunks = []
    current_chunk = []
    window_embedding = np.zeros(embeddings[0].shape)
    
    for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
        if current_chunk:
            similarity = np.dot(window_embedding, emb)
            position_weight = 1.2 - (i/len(sentences))  # Favor early/middle content
            if similarity * position_weight < CHUNK_SIMILARITY_THRESHOLD:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                window_embedding = np.zeros_like(emb)
        
        current_chunk.append(sent)
        window_embedding = (window_embedding + emb) / 2
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_document_summary(chunks: List[str]) -> str:
    """Hierarchical document summarization"""
    try:
        response = ollama.generate(
            model='mistral',
            prompt=f"Create a comprehensive document summary from these key sections:\n" + 
                   "\n".join([f"Section {i+1}: {chunk[:500]}" for i, chunk in enumerate(chunks[:8])]) +
                   "\nFocus on:\n- Core themes\n- Key entities\n- Major conclusions",
            options={'temperature': 0.2}
        )
        return response['response'].strip()
    except Exception as e:
        st.error(f"Summary error: {str(e)}")
        return "Document summary unavailable"

def create_contextual_chunks(chunks: List[str], doc_summary: str) -> List[str]:
    """Create chunks with document-aware context"""
    contextual_chunks = []
    for idx, chunk in enumerate(chunks):
        try:
            summary = ollama.generate(
                model='mistral',
                prompt=f"Document Context: {doc_summary}\n\n" +
                       f"Create a context-rich summary for this section:\n{chunk}\n" +
                       "Include:\n- Relation to main themes\n- Key details\n- Important context",
                options={'temperature': 0.1}
            )['response'].strip()
            contextual_chunks.append(
                f"## Section {idx+1}\n**Summary:** {summary}\n\n**Content:** {chunk}"
            )
        except:
            contextual_chunks.append(f"## Section {idx+1}\n{chunk}")
    return contextual_chunks

# Search system
def create_hybrid_index(chunks: List[str]):
    """Create optimized search indexes"""
    content = [re.sub(r'##.*\n', '', chunk) for chunk in chunks]  # Remove section markers
    embeddings = get_embeddings(content)
    
    # FAISS Index
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.add(embeddings.astype('float32'))
    
    # BM25 Index
    tokenized_chunks = [TOKENIZER.tokenize(text) for text in content]
    bm25_index = BM25Okapi(tokenized_chunks, k1=1.6, b=0.75)
    
    return index, bm25_index, embeddings

def hybrid_search(query: str, faiss_index, bm25_index, chunks: List[str]):
    """Context-aware hybrid search pipeline"""
    # First-stage retrieval
    query_embed = get_embeddings([query])[0]
    _, v_indices = faiss_index.search(np.array([query_embed]).astype('float32'), NUM_RERANK)
    
    tokenized_query = TOKENIZER.tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    b_indices = np.argsort(bm25_scores)[-NUM_RERANK:][::-1]
    
    # Combine candidates
    candidates = list(set(v_indices[0].tolist() + b_indices.tolist()))
    
    # Cross-encoder reranking
    pairs = [[query, re.sub(r'##.*\n', '', chunks[idx])] for idx in candidates]
    scores = cross_model.predict(pairs)
    
    results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return results[:10]

# UI Components
def doc_upload():
    """Document upload and processing component"""
    file = st.file_uploader("Upload Document", type=['txt'], key='doc_upload')
    if file and st.session_state.get('processed_doc') != file.name:
        with st.spinner("Analyzing document..."):
            text = file.read().decode()[:MAX_DOC_SIZE]
            summary, chunks = process_document(text)
            faiss_index, bm25_index, _ = create_hybrid_index(chunks)
            
            st.session_state.update({
                'processed_doc': file.name,
                'doc_summary': summary,
                'chunks': chunks,
                'faiss_index': faiss_index,
                'bm25_index': bm25_index
            })

def query_interface():
    """Main query and display interface"""
    query = st.text_input("Ask about the document:", key='doc_query')
    if query and 'chunks' in st.session_state:
        with st.spinner("Searching..."):
            results = hybrid_search(
                query,
                st.session_state['faiss_index'],
                st.session_state['bm25_index'],
                st.session_state['chunks']
            )
        
        with st.spinner("Generating answer..."):
            context = "\n".join([st.session_state['chunks'][idx] for idx, _ in results[:3]])
            response = ollama.generate(
                model='mistral',
                prompt=f"Document Summary: {st.session_state['doc_summary']}\n\n" +
                       f"Relevant Context:\n{context}\n\nQuestion: {query}\nAnswer:",
                options={'temperature': 0.3, 'max_tokens': 1000}
            )
            
            st.subheader("Answer")
            st.markdown(f"```\n{response['response']}\n```")
            
            st.subheader("Source Sections")
            for idx, score in results[:5]:
                chunk = st.session_state['chunks'][idx]
                st.markdown(f"**Section Score:** {score:.2f}")
                st.markdown(chunk)
                st.divider()

# Main app
def main():
    st.title("Document Context RAG")
    st.markdown("### Full-document aware Question Answering")
    
    doc_upload()
    if 'doc_summary' in st.session_state:
        with st.expander("Document Overview"):
            st.markdown(st.session_state['doc_summary'])
    
    query_interface()

if __name__ == "__main__":
    main()
