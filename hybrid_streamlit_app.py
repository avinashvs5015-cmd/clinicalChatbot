"""
Hybrid RAG Streamlit Application
Combines exact text matching with vector similarity search for document Q&A
"""

import streamlit as st
import os
import re
from pathlib import Path
import sys
from openai import OpenAI

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from hybrid_rag import HybridRAG

# Page configuration
st.set_page_config(
    page_title="Simple RAG Chat with NVIDIA Llama",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        margin-right: 2rem;
    }
    
    /* Status and info boxes */
    .info-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* References section styling */
    .references-section {
        background-color: #f5f5f5;
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 4px solid #757575;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        font-family: 'Courier New', monospace;
    }
    
    .references-title {
        font-weight: bold;
        color: #424242;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Loading spinner customization */
    .stSpinner > div {
        border-top-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_nvidia_client(api_key):
    """Initialize NVIDIA API client with OpenAI interface"""
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def format_references(search_results):
    """
    Format search results into a clean references section with unique files
    
    Args:
        search_results: List of search result dictionaries with metadata
        
    Returns:
        Formatted references string or empty string if no results
    """
    if not search_results:
        return ""
    
    # Group results by file and collect only meaningful reference info
    file_groups = {}
    
    for result in search_results:
        metadata = result.get('metadata', {})
        
        # Get file path from multiple possible metadata fields
        file_path = (metadata.get('file_path') or 
                    metadata.get('source_document') or 
                    'Unknown file')
        
        # Extract clean filename
        filename = os.path.basename(file_path) if file_path != 'Unknown file' else 'Unknown file'
        
        if filename not in file_groups:
            file_groups[filename] = {
                'pages': set(),
                'sections': set(),
                'max_confidence': 0
            }
        
        # Only add meaningful location information
        if 'page_number' in metadata and metadata['page_number'] is not None:
            file_groups[filename]['pages'].add(int(metadata['page_number']))
        
        # Track highest confidence for this file
        if 'combined_score' in result:
            confidence = result['combined_score']
            if confidence > file_groups[filename]['max_confidence']:
                file_groups[filename]['max_confidence'] = confidence
    
    # Format clean references
    references = []
    for i, (filename, group) in enumerate(file_groups.items(), 1):
        ref_parts = [filename]
        
        # Add page information if available
        if group['pages']:
            sorted_pages = sorted(group['pages'])
            if len(sorted_pages) == 1:
                ref_parts.append(f"Page {sorted_pages[0]}")
            elif len(sorted_pages) <= 3:
                ref_parts.append(f"Pages {', '.join(map(str, sorted_pages))}")
            else:
                ref_parts.append(f"Pages {sorted_pages[0]}-{sorted_pages[-1]} (and others)")
        
        references.append(" • ".join(ref_parts))
    
    if references:
        return "\n\n**References:**\n" + "\n".join(references)
    return ""

def clean_response_text(text):
    """
    Clean response text by removing markdown formatting and thinking patterns
    
    Args:
        text: Raw response text from the AI model
        
    Returns:
        Cleaned text suitable for display
    """
    if not text:
        return text
    
    # Remove thinking tags and patterns
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reasoning>.*?</reasoning>',
        r'<analysis>.*?</analysis>'
    ]
    
    cleaned = text
    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove markdown formatting for cleaner display
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Bold
    cleaned = re.sub(r'__(.*?)__', r'\1', cleaned)      # Bold
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Italic
    cleaned = re.sub(r'_(.*?)_', r'\1', cleaned)        # Italic
    cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)        # Code
    
    # Clean up whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned if cleaned else text

def generate_response(client, query, context="", model="nvidia/llama-3.3-nemotron-super-49b-v1.5", 
                     conversation_history=None, show_references=True):
    """
    Generate AI response using NVIDIA Llama API
    
    Args:
        client: OpenAI client configured for NVIDIA
        query: User question
        context: Retrieved document context
        model: Model name to use
        conversation_history: Previous conversation messages
        show_references: Whether to include references in response
        
    Returns:
        Generated response text
    """
    try:
        # Create system prompt
        reference_instruction = "" if show_references else "\n8. Do NOT include any 'References:', 'Sources:', or citation information in your response"
        
        system_prompt = f"""You are a helpful assistant that provides accurate answers based strictly on the provided document content.

INSTRUCTIONS:
1. Use ONLY information from the provided documents
2. Provide clear, comprehensive explanations when relevant information is available
3. If the documents contain relevant information, explain it thoroughly but concisely
4. Write in a conversational, easy-to-understand manner
5. If information is not in the documents, clearly state this
6. Use plain text without markdown formatting
7. Focus on answering the exact question asked{reference_instruction}

Your goal is to provide accurate, document-based answers that directly address the user's question."""

        # Prepare the prompt
        if context and context.strip():
            clean_context = context.replace("Context 1:", "").replace("Context 2:", "").replace("Context 3:", "")
            clean_context = re.sub(r'---+', '', clean_context).strip()
            
            user_prompt = f"""Based on the following document content, please answer the question:

DOCUMENT CONTENT:
{clean_context}

QUESTION: {query}

Please provide your answer based only on the information above."""
        else:
            user_prompt = f"I don't have any relevant document content to answer this question: {query}\n\nPlease let me know if you have specific documents you'd like me to search through."

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 4 exchanges)
        if conversation_history:
            for msg in conversation_history[-8:]:  # Last 4 exchanges (8 messages)
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query
        messages.append({"role": "user", "content": user_prompt})
        
        # Generate response
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent responses
            max_tokens=2048,
            stream=False
        )
        
        response = completion.choices[0].message.content
        return clean_response_text(response)
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the hybrid RAG system"""
    try:
        with st.spinner("Initializing RAG system with persistent embeddings..."):
            rag = HybridRAG()
            return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def display_system_status(rag_system):
    """Display system status information in the sidebar"""
    if rag_system:
        try:
            embedding_count = rag_system.vector_store.get_collection_count()
            text_chunk_count = len(rag_system.text_chunks)
            
            st.sidebar.markdown("""
            <div class="sidebar-info">
                <h4>System Status</h4>
                <p><strong>Status:</strong> Ready</p>
                <p><strong>Vector Embeddings:</strong> {}</p>
                <p><strong>Text Chunks:</strong> {}</p>
                <p><strong>Search Mode:</strong> Hybrid (Exact + Semantic)</p>
            </div>
            """.format(embedding_count, text_chunk_count), unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Status error: {e}")
    else:
        st.sidebar.error("System not initialized")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">Simple RAG Chat with NVIDIA Llama</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if not st.session_state.system_initialized:
        rag_system = initialize_rag_system()
        if rag_system:
            st.session_state.rag_system = rag_system
            st.session_state.system_initialized = True
            st.success("RAG system initialized successfully!")
        else:
            st.error("Failed to initialize RAG system. Please check your setup.")
            return
    else:
        rag_system = st.session_state.rag_system
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # References toggle
        show_references = st.checkbox(
            "Show References",
            value=True,
            help="Include source file information at the end of responses"
        )
        
        # Example questions
        st.markdown("### Try These Questions")
        example_questions = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "What does IPCC say about climate change?",
            "Who is Andrew of Padua?",
            "Tell me about John Galt's works"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}", use_container_width=True):
                # Add the question to chat and trigger processing
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.markdown("---")
        # Clear chat
        if st.button("Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
    
    # Configuration - Use secrets for API key
    try:
        api_key = st.secrets["NVIDIA_API_KEY"]
    except KeyError:
        st.error("NVIDIA API key not found in secrets. Please configure it in Streamlit Cloud.")
        st.stop()
    
    model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    num_results = 3  # Default number of document sections to retrieve
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">'
                f'<strong>You:</strong> {message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">'
                f'<strong>Assistant:</strong> {message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("NVIDIA API key not configured!")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
    
    # Process response if there's a pending user message
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user"):
        
        last_user_message = st.session_state.messages[-1]["content"]
        
        # Initialize client
        client = initialize_nvidia_client(api_key)
        
        # Search documents
        with st.spinner():
            try:
                context = rag_system.get_context(
                    last_user_message, 
                    search_type="hybrid", 
                    top_k=num_results
                )
                
                search_results = rag_system.search(
                    last_user_message, 
                    search_type="hybrid", 
                    top_k=num_results
                )
                    
            except Exception as e:
                st.error(f"Search error: {e}")
                context = ""
                search_results = []
        
        # Generate response
        with st.spinner("Generating response..."):
            try:
                # Get conversation history (excluding current message)
                conversation_history = st.session_state.messages[:-1]
                
                response = generate_response(
                    client=client,
                    query=last_user_message,
                    context=context,
                    model=model,
                    conversation_history=conversation_history,
                    show_references=show_references
                )
                
                # Add references if enabled
                if show_references and search_results:
                    references = format_references(search_results)
                    if references:
                        response += references
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Response generation error: {e}")

if __name__ == "__main__":
    main()