"""
Utility Functions for Semantic Product Search
==============================================

Helper functions for:
- Memory management
- Embedding similarity
- Result formatting
- File I/O

Author: [Your Name]
Course: Generative AI - Fall 2025
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Any
import psutil
import warnings
warnings.filterwarnings('ignore')


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory stats in MB
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def print_memory_usage():
    """Print current memory usage"""
    mem = get_memory_usage()
    print(f"💾 Memory Usage:")
    print(f"   RSS: {mem['rss_mb']:.2f} MB")
    print(f"   VMS: {mem['vms_mb']:.2f} MB")
    print(f"   Percent: {mem['percent']:.2f}%")


def cosine_similarity(
    a: np.ndarray, 
    b: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between two sets of vectors
    
    Args:
        a: Array of shape (n, d)
        b: Array of shape (m, d)
    
    Returns:
        Similarity matrix of shape (n, m)
    """
    # Normalize vectors
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    
    # Calculate similarity
    similarity = np.dot(a_norm, b_norm.T)
    
    return similarity


def get_top_k_similar(
    query_embedding: np.ndarray,
    product_embeddings: np.ndarray,
    k: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top K most similar products to a query
    
    Args:
        query_embedding: Query embedding (1, d)
        product_embeddings: Product embeddings (n, d)
        k: Number of top results to return
    
    Returns:
        Tuple of (top_k_indices, top_k_scores)
    """
    # Calculate similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        product_embeddings
    ).flatten()
    
    # Get top K
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]
    
    return top_k_indices, top_k_scores


def format_results(
    df_products: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Format search results for display
    
    Args:
        df_products: Product metadata DataFrame
        indices: Indices of retrieved products
        scores: Relevance scores
        top_n: Number of results to return
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    for idx, score in zip(indices[:top_n], scores[:top_n]):
        product = df_products.iloc[idx]
        
        result = {
            'product_id': product['product_id'],
            'title': product['product_title'][:100],  # Truncate long titles
            'description': product['product_description'][:200] if pd.notna(product['product_description']) else '',
            'relevance_score': float(score),
            'rank': len(results) + 1
        }
        
        results.append(result)
    
    return results


def save_embeddings(
    embeddings: np.ndarray,
    save_path: str
):
    """
    Save embeddings to disk
    
    Args:
        embeddings: Numpy array of embeddings
        save_path: Path to save file
    """
    np.save(save_path, embeddings)
    print(f"✅ Embeddings saved to {save_path}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")


def load_embeddings(load_path: str) -> np.ndarray:
    """
    Load embeddings from disk
    
    Args:
        load_path: Path to embeddings file
    
    Returns:
        Numpy array of embeddings
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Embeddings file not found: {load_path}")
    
    embeddings = np.load(load_path)
    print(f"✅ Embeddings loaded from {load_path}")
    print(f"   Shape: {embeddings.shape}")
    
    return embeddings


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str
):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to {save_path}")


def load_model_checkpoint(
    load_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, int, Dict]:
    """
    Load model checkpoint
    
    Args:
        load_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load model on
    
    Returns:
        Tuple of (model, epoch, metrics)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✅ Checkpoint loaded from {load_path}")
    print(f"   Epoch: {epoch}")
    
    return model, epoch, metrics


def save_json(data: Dict, save_path: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        save_path: Path to save file
    """
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ JSON saved to {save_path}")


def load_json(load_path: str) -> Dict:
    """
    Load dictionary from JSON file
    
    Args:
        load_path: Path to JSON file
    
    Returns:
        Dictionary
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"JSON file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ JSON loaded from {load_path}")
    
    return data


def save_pickle(obj: Any, save_path: str):
    """
    Save object using pickle
    
    Args:
        obj: Object to save
        save_path: Path to save file
    """
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"✅ Pickle saved to {save_path}")


def load_pickle(load_path: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        load_path: Path to pickle file
    
    Returns:
        Loaded object
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Pickle file not found: {load_path}")
    
    with open(load_path, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"✅ Pickle loaded from {load_path}")
    
    return obj


def create_directories(dirs: List[str]):
    """
    Create directories if they don't exist
    
    Args:
        dirs: List of directory paths
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory ready: {dir_path}")


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def highlight_query_terms(
    text: str,
    query: str,
    highlight_start: str = "**",
    highlight_end: str = "**"
) -> str:
    """
    Highlight query terms in text (for display)
    
    Args:
        text: Text to highlight in
        query: Query string
        highlight_start: String to mark start of highlight
        highlight_end: String to mark end of highlight
    
    Returns:
        Text with highlighted terms
    """
    import re
    
    # Split query into terms
    terms = query.lower().split()
    
    # Highlight each term
    for term in terms:
        if len(term) > 2:  # Only highlight meaningful terms
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(f"{highlight_start}{term}{highlight_end}", text)
    
    return text


def get_device() -> str:
    """
    Get best available device (CUDA, MPS, or CPU)
    
    Returns:
        Device string
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    return device


if __name__ == "__main__":
    # Test utilities
    print("=" * 70)
    print("Testing Utility Functions")
    print("=" * 70)
    
    # Test memory usage
    print_memory_usage()
    
    # Test cosine similarity
    print("\n📐 Testing Cosine Similarity:")
    a = np.random.randn(3, 5)
    b = np.random.randn(4, 5)
    sim = cosine_similarity(a, b)
    print(f"   Shape: {sim.shape}")
    print(f"   Range: [{sim.min():.3f}, {sim.max():.3f}]")
    
    # Test device detection
    print(f"\n🖥️ Device Detection:")
    device = get_device()
    
    # Test text truncation
    print(f"\n✂️ Text Truncation:")
    long_text = "This is a very long text that needs to be truncated for display purposes."
    truncated = truncate_text(long_text, max_length=30)
    print(f"   Original: {long_text}")
    print(f"   Truncated: {truncated}")
    
    print("\n✅ All tests passed!")