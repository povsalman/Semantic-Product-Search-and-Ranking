"""
Neural Ranking Model for Semantic Product Search
=================================================

Implements the ranking model architecture based on DistilBERT
for predicting relevance scores between queries and products.

Author: [Your Name]
Course: Generative AI - Fall 2025
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SemanticRankingModel(nn.Module):
    """
    Semantic ranking model for query-product relevance prediction
    
    Architecture:
    - DistilBERT encoder (or any HuggingFace model)
    - Dropout for regularization
    - Linear layer for regression
    - Sigmoid activation for [0,1] relevance scores
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        """
        Initialize the ranking model
        
        Args:
            model_name: HuggingFace model name/path
            dropout: Dropout probability
            freeze_encoder: Whether to freeze BERT parameters
        """
        super(SemanticRankingModel, self).__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        
        # Freeze encoder if specified (faster training, less memory)
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get hidden size from BERT config
        hidden_size = self.bert.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Regression head
        self.regressor = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
        
        Returns:
            Relevance scores [batch_size] in range [0, 1]
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        x = self.dropout(cls_embedding)
        
        # Regression to single relevance score
        relevance_score = self.regressor(x)
        
        # Sigmoid to constrain to [0, 1] range
        relevance_score = torch.sigmoid(relevance_score)
        
        # Squeeze to remove last dimension
        return relevance_score.squeeze(-1)
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get BERT embeddings without regression head
        
        Useful for visualization and analysis
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            [CLS] embeddings [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding


class BiEncoderRankingModel(nn.Module):
    """
    Bi-encoder architecture for faster inference
    
    Encodes queries and products separately, then combines for ranking.
    More scalable for large product catalogs.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        """
        Initialize bi-encoder model
        
        Args:
            model_name: Sentence transformer model name
            hidden_dim: Hidden dimension for ranking network
            dropout: Dropout probability
        """
        super(BiEncoderRankingModel, self).__init__()
        
        from sentence_transformers import SentenceTransformer
        
        # Load sentence transformer
        self.encoder = SentenceTransformer(model_name)
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Freeze encoder (we use pretrained embeddings)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Ranking network
        # Input: [query_emb || product_emb || element-wise product]
        input_dim = embedding_dim * 3
        
        self.ranking_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries to embeddings"""
        embeddings = self.encoder.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings
    
    def encode_products(self, products: List[str]) -> torch.Tensor:
        """Encode products to embeddings"""
        embeddings = self.encoder.encode(
            products,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        product_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for bi-encoder
        
        Args:
            query_embeddings: Query embeddings [batch_size, embedding_dim]
            product_embeddings: Product embeddings [batch_size, embedding_dim]
        
        Returns:
            Relevance scores [batch_size]
        """
        # Element-wise product (interaction feature)
        interaction = query_embeddings * product_embeddings
        
        # Concatenate: [query, product, interaction]
        combined = torch.cat([
            query_embeddings,
            product_embeddings,
            interaction
        ], dim=1)
        
        # Ranking network
        scores = self.ranking_net(combined)
        
        return scores.squeeze(-1)


def load_trained_model(
    model_path: str,
    model_name: str = "distilbert-base-uncased",
    device: str = "cpu"
) -> SemanticRankingModel:
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to saved model checkpoint (.pth file)
        model_name: Name of the base BERT model used
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model ready for inference
    """
    # Initialize model
    model = SemanticRankingModel(model_name)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    # Move to device
    model = model.to(device)
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {device}")
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


if __name__ == "__main__":
    # Test the model
    print("=" * 70)
    print("Testing SemanticRankingModel")
    print("=" * 70)
    
    # Initialize model
    model = SemanticRankingModel("distilbert-base-uncased")
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / 1e6:.2f} MB")
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)
    
    with torch.no_grad():
        scores = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input_ids.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Output scores: {scores}")
    print(f"Scores in [0,1]: {torch.all((scores >= 0) & (scores <= 1))}")