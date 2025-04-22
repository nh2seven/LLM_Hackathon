import os
import glob
import json
import logging
import requests
from typing import List, Dict, Any, Optional
import argparse
from simple_rag.vector_store import TfidfVectorStore, Document
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_DIR = "knowledge_base_vector_store"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = "mistral:latest"  # Changed from mistral:7b to mistral:latest
MAX_TOKENS = 4000  # Chunk size for long documents

class OllamaLLM:
    """Simple wrapper for Ollama API"""
    
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.base_url = OLLAMA_BASE_URL
        self.api_url = OLLAMA_API_URL
        self.is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama API is available"""
        try:
            # First check if Ollama server is running at all
            response = requests.get(f"{self.base_url}")
            if response.status_code != 200:
                logger.warning(f"Ollama server not responding at {self.base_url}. Status: {response.status_code}")
                return False
                
            # Then check if models API is available
            models_response = requests.get(f"{self.base_url}/api/tags")
            if models_response.status_code != 200:
                logger.warning(f"Ollama models API not available. Status: {models_response.status_code}")
                return False
                
            # Check if our model is available
            available_models = models_response.json().get("models", [])
            model_names = [model.get("name") for model in available_models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found in available models: {model_names}")
                logger.warning(f"Please run: ollama pull {self.model_name}")
                return False
                
            logger.info(f"Ollama API is available with model {self.model_name}")
            return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to Ollama server at {self.base_url}")
            return False
        except Exception as e:
            logger.warning(f"Error checking Ollama availability: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM"""
        if not self.is_available:
            return self._fallback_generate(prompt, system_prompt)
            
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # Try the standard generate endpoint
            logger.info(f"Calling Ollama API with model {self.model_name}")
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 404:
                # If 404, try the chat completion endpoint as fallback
                logger.info("Generate endpoint not found, trying chat completion endpoint")
                chat_payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
                response = requests.post(f"{self.base_url}/api/chat", headers=headers, json=chat_payload)
            
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if "response" in result:
                return result["response"]
            elif "message" in result:
                return result["message"].get("content", "")
            else:
                logger.warning(f"Unexpected response format: {result}")
                return "Error: Unexpected response format from Ollama API"
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return self._fallback_generate(prompt, system_prompt)
    
    def _fallback_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Fallback method when Ollama is not available"""
        # Extract the topic or question from the prompt
        query = ""
        if "Question:" in prompt:
            query_part = prompt.split("Question:")[1].split("\n")[0].strip()
            query = query_part
        
        # Generate a simple response based on the query
        if "overview" in prompt.lower() or "introduction" in prompt.lower():
            return f"This is an automatically generated overview about {query}. Since Ollama API is not available, this is placeholder content. The overview would typically describe key concepts and importance of the topic."
        
        if "summary" in prompt.lower() or "conclusion" in prompt.lower():
            return f"This is an automatically generated summary for {query}. Since Ollama API is not available, this is placeholder content. The summary would typically include key takeaways and applications."
        
        if "section" in prompt.lower() or "subtopic" in prompt.lower():
            section = ""
            if "about '" in prompt:
                section = prompt.split("about '")[1].split("'")[0]
            
            return f"This is automatically generated content about '{section or query}'. Since Ollama API is not available, this is placeholder content. This section would typically include detailed explanations, examples, and key concepts."
        
        if "caption" in prompt.lower() or "image" in prompt.lower():
            return f"This is an automatically generated caption for an image related to {query}. Since Ollama API is not available, this is a placeholder caption."
        
        return f"This is automatically generated content related to your query. Since Ollama API is not available, this is placeholder content."

def clean_llm_response(response: str) -> str:
    """Clean up the LLM response to remove common fluff and context statements"""
    
    # Remove common prefixes used by LLMs
    prefixes_to_remove = [
        "Based on the provided context,",
        "Based on the context provided,",
        "According to the provided context,",
        "From the context provided,",
        "The provided context discusses",
        "The provided context shows",
        "The provided context is about",
        "The context mentions",
        "The context discusses",
        "In the provided context,",
        "From the provided information,",
        "Based on the information provided,",
        "The context provides",
        "Based on my analysis of the context,",
        "The context explains",
        "According to the context,",
        "In the context,",
        "As per the context,",
        "Looking at the context provided,",
        "Analyzing the context provided,",
    ]
    
    # Try to find and remove these prefixes
    cleaned_response = response
    for prefix in prefixes_to_remove:
        if cleaned_response.startswith(prefix):
            cleaned_response = cleaned_response[len(prefix):].lstrip()
            break
    
    # Remove any sentences that reference "context" or "provided information"
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
    filtered_sentences = []
    
    context_patterns = [
        r'context',
        r'provided information',
        r'provided document',
        r'I don\'t have enough information',
        r'based on the information I have',
        r'the information provided',
    ]
    
    for sentence in sentences:
        if not any(re.search(pattern, sentence, re.IGNORECASE) for pattern in context_patterns):
            filtered_sentences.append(sentence)
    
    # Ensure we don't make it empty if everything was filtered
    if filtered_sentences:
        cleaned_response = ' '.join(filtered_sentences)
    
    # Remove any remaining reference to answering the original question
    cleaned_response = re.sub(r'To answer your question[,:]?\s*', '', cleaned_response)
    
    return cleaned_response

def process_directory(data_dir: str) -> List[Document]:
    """Process text files in directory and return document objects"""
    logger.info(f"Processing text files in {data_dir}")
    documents = []
    
    # Get all .txt files recursively
    txt_files = glob.glob(f"{data_dir}/**/*.txt", recursive=True)
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "type": "text"
            }
            
            # Add document
            documents.append(Document(page_content=content, metadata=metadata))
            logger.info(f"Processed: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def process_topic_directory(topic_dir: str) -> List[Document]:
    """Process text files in a specific topic directory and return document objects"""
    logger.info(f"Processing text files for topic: {os.path.basename(topic_dir)}")
    documents = []
    
    # Get all .txt files in the topic directory
    txt_files = glob.glob(f"{topic_dir}/**/*.txt", recursive=True)
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "type": "text",
                "topic": os.path.basename(topic_dir)
            }
            
            # Add document
            documents.append(Document(page_content=content, metadata=metadata))
            logger.info(f"Processed: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = MAX_TOKENS) -> List[Document]:
    """Split documents into smaller chunks if they are too long"""
    logger.info("Chunking documents...")
    chunked_docs = []
    
    for doc in documents:
        content = doc.page_content
        
        # If content is short enough, keep as is
        if len(content) <= chunk_size:
            chunked_docs.append(doc)
            continue
        
        # Split into chunks by paragraphs
        paragraphs = content.split('\n\n')
        current_chunk = ""
        current_metadata = doc.metadata.copy()
        
        for i, para in enumerate(paragraphs):
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk
                if current_chunk:
                    current_metadata["chunk"] = len(chunked_docs)
                    chunked_docs.append(Document(page_content=current_chunk, metadata=current_metadata))
                
                # Start new chunk
                current_chunk = para + "\n\n"
        
        # Add the last chunk if it's not empty
        if current_chunk:
            current_metadata["chunk"] = len(chunked_docs)
            chunked_docs.append(Document(page_content=current_chunk, metadata=current_metadata))
    
    logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs

def create_vector_store(chunks: List[Document]) -> TfidfVectorStore:
    """Create and save TF-IDF vector store from document chunks"""
    logger.info("Creating TF-IDF vector store...")
    vector_store = TfidfVectorStore().add_documents(chunks)
    
    # Save the vector store for future use
    vector_store.save_local(VECTOR_STORE_DIR)
    return vector_store

def create_topic_vector_store(topic_dir: str) -> TfidfVectorStore:
    """Create a vector store specifically for a single topic"""
    topic_name = os.path.basename(topic_dir)
    logger.info(f"Creating vector store for topic: {topic_name}")
    
    # Process documents from the topic directory
    documents = process_topic_directory(topic_dir)
    logger.info(f"Processed {len(documents)} documents for topic {topic_name}")
    
    if not documents:
        logger.warning(f"No documents found for topic {topic_name}")
        # Return empty vector store if no documents found
        return TfidfVectorStore().add_documents([])
    
    # Chunk documents if needed
    chunks = chunk_documents(documents)
    
    # Create the vector store
    vector_store = TfidfVectorStore().add_documents(chunks)
    logger.info(f"Created vector store with {len(chunks)} chunks for topic {topic_name}")
    
    return vector_store

def load_or_create_vector_store(data_dir: str) -> TfidfVectorStore:
    """Load existing vector store or create a new one"""
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_DIR) and all(
        os.path.exists(f"{VECTOR_STORE_DIR}/{file}")
        for file in ["vectorizer.pkl", "tfidf_matrix.pkl", "documents.pkl"]
    ):
        # Load existing vector store
        try:
            vector_store = TfidfVectorStore.load_local(VECTOR_STORE_DIR)
            logger.info("Loaded existing vector store")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.info("Creating new vector store...")
    
    # Process documents and create vector store
    logger.info("Processing documents...")
    documents = process_directory(data_dir)
    logger.info(f"Processed {len(documents)} documents")
    
    chunks = chunk_documents(documents)
    return create_vector_store(chunks)

def format_rag_prompt(query: str, context_docs: List[Document]) -> Dict[str, str]:
    """Format prompt with retrieval results for RAG"""
    context_text = "\n\n".join([
        f"Document from {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}"
        for doc in context_docs
    ])
    
    system_prompt = """You are a helpful AI assistant creating educational lecture notes.
When responding, focus ONLY on providing clear, concise, and well-structured content.
DO NOT mention the context, sources, or make any meta-references to the information provided.
DO NOT use phrases like "based on the context" or "the provided information shows."
DO NOT apologize or mention limitations of your knowledge.
Simply present the information as factual, educational content in a professional tone.
If you don't know something, just omit that section rather than drawing attention to missing information."""
    
    user_prompt = f"""Please generate educational content for the following topic using the references provided.
Format your response as clear lecture notes without any meta-references to sources or context.

Topic: {query}

References:
{context_text}

Your response should be well-structured, educational content ONLY, with no references to these sources or to "context" or "provided information"."""
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }

def rag_query(query: str, vector_store: TfidfVectorStore, llm: OllamaLLM, top_k: int = 4) -> str:
    """Execute a RAG query using the vector store and LLM"""
    # Retrieve relevant documents
    relevant_docs = vector_store.get_relevant_documents(query, k=top_k)
    
    if not relevant_docs:
        return "I couldn't find any relevant information to answer your question."
    
    # Format prompt with context
    prompts = format_rag_prompt(query, relevant_docs)
    
    # Generate response using LLM
    response = llm.generate(prompts["user_prompt"], system_prompt=prompts["system_prompt"])
    
    # Clean the response
    cleaned_response = clean_llm_response(response)
    
    return cleaned_response

def main():
    parser = argparse.ArgumentParser(description="RAG Application using Ollama and TF-IDF")
    parser.add_argument("--data_dir", type=str, default="LLM_DATASET", 
                        help="Directory containing text files for knowledge base")
    parser.add_argument("--query", type=str, help="Query to run in non-interactive mode")
    args = parser.parse_args()
    
    # Load or create vector store
    vector_store = load_or_create_vector_store(args.data_dir)
    
    # Initialize LLM
    llm = OllamaLLM()
    
    if args.query:
        # Run a single query
        response = rag_query(args.query, vector_store, llm)
        print(f"\nQuery: {args.query}")
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        print("\nRAG Application with Ollama and TF-IDF Vector Store")
        print("Type 'exit' or 'quit' to exit\n")
        
        while True:
            query = input("\nEnter your question: ")
            
            if query.lower() in ['exit', 'quit']:
                break
            
            response = rag_query(query, vector_store, llm)
            print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()