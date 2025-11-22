"""
Semantic Product Search Web Application
========================================

Streamlit web interface for real-time product search using trained
semantic ranking model.

Author: [Your Name]
Course: Generative AI - Fall 2025

Usage:
    streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append('src')

from model import SemanticRankingModel
from preprocessor import MinimalPreprocessor
from utils import (
    cosine_similarity,
    get_top_k_similar,
    format_results,
    load_embeddings
)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration"""
    MODEL_PATH = "models/best_model.pth"
    EMBEDDINGS_PATH = "models/product_embeddings.npy"
    PRODUCTS_PATH = "models/product_metadata.parquet"
    
    MODEL_NAME = "distilbert-base-uncased"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_LENGTH = 128
    
    TOP_K_RETRIEVAL = 100  # Retrieve top 100 candidates
    TOP_N_DISPLAY = 10      # Display top 10 results

config = Config()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Semantic Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .result-score {
        font-size: 0.9rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Load Models and Data (Cached)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models and data (cached for performance)"""
    
    with st.spinner("🔄 Loading models... This may take a minute..."):
        # Check if required files exist
        required_files = [
            config.MODEL_PATH,
            config.EMBEDDINGS_PATH,
            config.PRODUCTS_PATH
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"❌ Missing files: {', '.join(missing_files)}")
            st.info("""
            Please ensure you have:
            1. Trained the model on Kaggle
            2. Downloaded the model files
            3. Placed them in the 'models/' directory
            
            Required files:
            - models/best_model.pth
            - models/product_embeddings.npy
            - models/product_metadata.parquet
            """)
            st.stop()
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load ranking model
        model = SemanticRankingModel(config.MODEL_NAME)
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Load embedding model (for query encoding)
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        embedding_model = embedding_model.to(device)
        
        # Load product embeddings
        product_embeddings = load_embeddings(config.EMBEDDINGS_PATH)
        
        # Load product metadata
        df_products = pd.read_parquet(config.PRODUCTS_PATH)
        
        # Preprocessor
        preprocessor = MinimalPreprocessor()
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'embedding_model': embedding_model,
            'product_embeddings': product_embeddings,
            'df_products': df_products,
            'preprocessor': preprocessor,
            'device': device
        }

# ============================================================================
# Search Functions
# ============================================================================

def search_products(query: str, resources: dict, top_n: int = 10) -> dict:
    """
    Perform semantic search for products
    
    Args:
        query: Search query
        resources: Dictionary of loaded models and data
        top_n: Number of results to return
    
    Returns:
        Dictionary with results and metrics
    """
    start_time = time.time()
    
    # Extract resources
    model = resources['model']
    tokenizer = resources['tokenizer']
    embedding_model = resources['embedding_model']
    product_embeddings = resources['product_embeddings']
    df_products = resources['df_products']
    preprocessor = resources['preprocessor']
    device = resources['device']
    
    # 1. Encode query
    query_embedding = embedding_model.encode(
        [query],
        convert_to_tensor=False,
        show_progress_bar=False
    )
    
    # 2. Retrieve top-K candidates using cosine similarity
    top_k_indices, _ = get_top_k_similar(
        query_embedding,
        product_embeddings,
        k=config.TOP_K_RETRIEVAL
    )
    
    # 3. Re-rank candidates using trained model
    candidate_products = df_products.iloc[top_k_indices]
    
    # Prepare inputs for ranking model
    queries_batch = [query] * len(candidate_products)
    products_batch = candidate_products['product_text'].tolist()
    
    # Tokenize
    inputs = tokenizer(
        queries_batch,
        products_batch,
        add_special_tokens=True,
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
        predictions = predictions.cpu().numpy()
    
    # 4. Sort by predicted relevance
    sorted_indices = np.argsort(predictions)[::-1]
    top_indices = top_k_indices[sorted_indices[:top_n]]
    top_scores = predictions[sorted_indices[:top_n]]
    
    # 5. Format results
    results = format_results(
        df_products,
        top_indices,
        top_scores,
        top_n=top_n
    )
    
    end_time = time.time()
    
    return {
        'results': results,
        'num_results': len(results),
        'search_time': end_time - start_time,
        'num_candidates': len(candidate_products)
    }

# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Semantic Product Search</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Powered by Deep Learning & Transformer Models</p>',
        unsafe_allow_html=True
    )
    
    # Load resources
    try:
        resources = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_n = st.slider(
            "Number of Results",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        st.divider()
        
        st.subheader("📊 Model Info")
        st.info(f"""
        **Ranking Model:** {config.MODEL_NAME}
        
        **Embedding Model:** {config.EMBEDDING_MODEL.split('/')[-1]}
        
        **Device:** {resources['device'].upper()}
        
        **Products Indexed:** {len(resources['df_products']):,}
        """)
        
        st.divider()
        
        st.subheader("📖 About")
        st.markdown("""
        This application uses a **semantic search** approach to find
        relevant products based on natural language queries.
        
        **Features:**
        - 🧠 Deep learning-based ranking
        - 🔤 Natural language understanding
        - ⚡ Real-time search
        - 📊 Relevance scoring
        """)
        
        st.divider()
        
        st.caption("Task 2: Semantic Product Search")
        st.caption("Generative AI - Fall 2025")
        st.caption("NUCES Islamabad")
    
    # Main content
    st.subheader("🔎 Search for Products")
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "",
            placeholder="Enter your search query (e.g., 'wireless bluetooth headphones with noise cancellation')",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Example queries
    st.caption("💡 **Try these:** wireless headphones | gaming laptop | running shoes | kitchen appliances")
    
    # Perform search
    if search_button and query:
        with st.spinner("🔄 Searching..."):
            try:
                # Perform search
                search_results = search_products(query, resources, top_n=top_n)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{search_results['num_results']}</h3>
                        <p>Results Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{search_results['search_time']:.2f}s</h3>
                        <p>Search Time</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{search_results['num_candidates']}</h3>
                        <p>Candidates Ranked</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Display results
                st.subheader("📦 Top Results")
                
                for result in search_results['results']:
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">
                                {result['rank']}. {result['title']}
                            </div>
                            <p style="color: #666; margin-bottom: 0.5rem;">
                                {result['description'][:200]}{"..." if len(result['description']) > 200 else ""}
                            </p>
                            <div class="result-score">
                                ⭐ Relevance Score: {result['relevance_score']:.3f}
                            </div>
                            <p style="color: #999; font-size: 0.85rem; margin-top: 0.5rem;">
                                Product ID: {result['product_id']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
                st.exception(e)
    
    elif search_button:
        st.warning("⚠️ Please enter a search query")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.9rem;">
        <p>Built with Streamlit & PyTorch | Task 2: Semantic Product Search</p>
        <p>Course: Generative AI (Fall 2025) | Instructor: Dr. Akhtar Jamil</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()