import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from io import StringIO
import PyPDF2

def read_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Set up page config
st.set_page_config(
    page_title="Streamlit Ollama Chatbot with RAG",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize embedding model
embedding_model = "nomic-embed-text"


def fetch_models_from_server():
    """
    Fetch available models from Ollama server using the API.
    """
    try:
        models_info = ollama.list()
        print(models_info)
        available_models = [model["name"] for model in models_info if model.get("available")]
        return available_models
    except Exception as e:
        st.error(f"Failed to fetch models from Ollama server: {e}")
        return []

def create_vector_store(chunks, embedding_model="nomic-embed-text"):
    """
    Create a vector store using the embedding model from Ollama server.
    """
    try:
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embeddings for each chunk using Ollama's embedding model
                response = ollama.embed(model=embedding_model, input=chunk)
                embedding_vector = response.get("embeddings")

                if embedding_vector:
                    embeddings.append(embedding_vector)
                else:
                    st.warning(f"Embedding generation failed for chunk {i}. Skipping this chunk.")
                    continue

            except Exception as e:
                st.error(f"Error generating embedding for chunk {i}: {e}")
                continue

        if not embeddings:
            st.error("No embeddings were generated. Check your document or embedding model.")
            st.stop()

        # Create an index for efficient retrieval
        index = {i: embeddings[i] for i in range(len(embeddings))}
        return index, embeddings
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()




# Document Chunking Function
def chunk_document(document: str, chunk_size: int = 768):
    """
    Chunk the document into smaller pieces of text.
    """
    words = document.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Retrieve relevant chunks based on user query
def retrieve_relevant_chunks(query, index, embeddings, chunks):
    """
    Retrieve relevant chunks based on query embeddings.
    """
    try:
        # Generate embedding for the query
        query_embedding = ollama.embed(model=embedding_model, input=query).get("embeddings")
        if not query_embedding:
            st.error("Failed to generate query embedding.")
            return []

        # Calculate similarity and retrieve top relevant chunks
        similarities = [
            (i, cosine_similarity(query_embedding, embedding))
            for i, embedding in index.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunks[i] for i, _ in similarities[:3]]  # Top 3 relevant chunks
        return top_chunks
    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {e}")
        return []

def extract_model_names(models_info: dict) -> tuple:
    """
    Extracts the model names from the models information.
    """
    return tuple(model["name"] for model in models_info["models"])
    
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors using NumPy."""
    # Ensure the vectors are numpy arrays
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # Return zero similarity if either vector is zero-length

    return dot_product / (norm_vec1 * norm_vec2)
    
def main():
    """
    The main function that runs the application.
    """
    st.subheader("Streamlit Ollama Chatbot", divider="red", anchor=False)

    # Radio button for selecting between normal chat and RAG mode
    mode = st.radio("Select Chat Mode:", ("Normal Chat", "RAG Chat"))

    if mode == "Normal Chat":
        # Fetch available models from Ollama
        models_info = ollama.list()
        available_models = extract_model_names(models_info)

        if available_models:
            selected_model = st.selectbox(
                "Pick a model available locally on your system ↓", available_models
            )

        else:
            st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
            st.page_link("https://ollama.com/library", label="Pull model(s) from Ollama", icon="🦙")

        message_container = st.container(height=500, border=True)

        # Initialize session state for storing messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Handle user input and model response for normal chat
        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                # Add user message to session state
                st.session_state.messages.append(
                    {"role": "user", "content": prompt})

                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Give me a moment..."):
                        # Generate response using Ollama model
                        response = ollama.chat(
                            model=selected_model,
                            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                            stream=True,  # Stream the response
                            temperature=0.1,  # Adjust temperature to control randomness
                            max_tokens=200    # Limit response length
                        )

                        response_text = ""
                        response_container = st.empty()  # Temporary container for streaming responses

                        # Accumulate and display the response as it streams
                        for chunk in response:
                            if chunk.get('message'):
                                response_text += chunk['message']['content']
                                response_container.markdown(response_text)

                        # Finalize and append to session state after streaming ends
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )

            except Exception as e:
                st.error(f"Error: {e}", icon="⛔️")
                
                
                
                
    # Define the RAG Chat mode
    elif mode == "RAG Chat":
            # Fetch models directly from Ollama server
            models_info = ollama.list()
            available_models = extract_model_names(models_info)

            # Upload document and process it
            document = st.file_uploader("Upload Document", type=["txt", "pdf"])

            if document is not None:
                try:
                    # Read and process the document based on file type
                    if document.type == "text/plain":
                        document_text = StringIO(document.getvalue().decode("utf-8")).read()
                    elif document.type == "application/pdf":
                        document_text = read_pdf(document)
                    else:
                        st.error("Only text (.txt) and PDF documents are supported for now.")
                        st.stop()

                    # Chunk the document and create a vector store with the specified embedding model
                    chunks = chunk_document(document_text)
                    index, embeddings = create_vector_store(chunks)

                    st.success("Document processed successfully. You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    st.stop()

                # Display chat history for RAG chat
                message_container = st.container()
                if available_models:
                    selected_model = st.selectbox("Select a model from Ollama server ↓", available_models)
                else:
                    st.warning("No models available on Ollama server!", icon="⚠️")
                    st.stop()

                # Initialize session state for storing messages in RAG mode
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat history
                for message in st.session_state.messages:
                    avatar = "🤖" if message["role"] == "assistant" else "😎"
                    with message_container.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])

                # Handle user input and model response in RAG mode
                if prompt := st.chat_input("Enter a prompt here..."):
                    try:
                        # Add user message to session state
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        message_container.chat_message("user", avatar="😎").markdown(prompt)

                        with message_container.chat_message("assistant", avatar="🤖"):
                            with st.spinner("Connecting to Ollama server..."):
                                # Retrieve relevant chunks from the vector store
                                relevant_chunks = retrieve_relevant_chunks(prompt, index, embeddings, chunks)

                                # Combine the relevant chunks into a single context
                                context = "\n".join(relevant_chunks)

                                # Send the context with the query to Ollama server for response generation
                                response = ollama.chat(
                                    model=selected_model,
                                    messages=[
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": context}
                                    ],
                                    stream=True
                                )

                                response_text = ""
                                response_container = st.empty()  # Temporary container for streaming responses

                                # Accumulate and display the response as it streams
                                for chunk in response:
                                    if chunk.get('message'):
                                        response_text += chunk['message']['content']
                                        response_container.markdown(response_text)

                                # Finalize and append to session state after streaming ends
                                st.session_state.messages.append({"role": "assistant", "content": response_text})

                    except Exception as e:
                        st.error(f"Error: {e}", icon="⛔️")        



if __name__ == "__main__":
    main()
