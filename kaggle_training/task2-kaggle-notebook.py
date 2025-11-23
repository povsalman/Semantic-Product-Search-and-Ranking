#!/usr/bin/env python
# coding: utf-8

# Semantic Product Search and Ranking - Kaggle Training Notebook
# =======================================================================
# 
# Complete training pipeline for semantic product search using BERT-based models.
# Optimized for Kaggle P100 GPU with 12-hour session limit.
# 
# - Author: Salman Khan
# - Course: Generative AI - Fall 2025
# - Institution: NUCES Islamabad
# 
# Instructions:
# 1. Upload dataset files to Kaggle (examples.parquet, products.parquet)
# 2. Enable GPU in settings (P100 or T4)
# 3. Enable Internet access
# 4. Run all cells sequentially
# 5. Download trained model files after completion

# ### Step 1: Install Required Packages

# - Purpose: Install all required Python libraries and pin critical versions.
# - Inputs: None (uses `pip` to fetch packages from PyPI).
# - Key Actions: Upgrade core transformer libraries; install `ranx`, `contractions`; pin `protobuf==3.20.3` to avoid serialization errors; suppress warnings.
# - Outputs: Console confirmation of successful installations; environment ready for subsequent steps.

# ============================================================================
# Step 1: Install Required Packages
# ============================================================================

# Install latest versions of main libraries
get_ipython().system('pip install -U -q transformers')
get_ipython().system('pip install -U -q sentence-transformers')
get_ipython().system('pip install -q ranx==0.3.16')
get_ipython().system('pip install -q contractions==0.1.73')

# FIX: Pin protobuf to 3.20.3 to fix the 'GetPrototype' and 'MessageFactory' errors
get_ipython().system('pip install -q protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore')

print("✅ Packages installed successfully!")


# ### Step 2: Import Libraries

# - Purpose: Import libraries, configure environment, set seeds for reproducibility.
# - Inputs: Environment variable `TF_CPP_MIN_LOG_LEVEL`; GPU availability; NLTK corpora.
# - Key Actions: Suppress noisy logs; import data/NLP/deep learning/evaluation libs; set random seeds; detect device and GPU stats; download NLTK resources.
# - Outputs: Device information printed; confirmation of successful imports.

# ============================================================================
# Step 2: Import Libraries
# ============================================================================

# 1. Suppress TensorFlow/CUDA warnings (Must be before other imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Standard libraries
import json
import time
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer

# Evaluation
from ranx import Qrels, Run, evaluate

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from tqdm.auto import tqdm

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)

print("✅ All libraries imported successfully!")


# ### Step 3: Configuration and Hyperparameters

# - Purpose: Centralize configuration (paths, model names, hyperparameters, splits).
# - Inputs: None (static class definition).
# - Key Settings: `MODEL_NAME`, `EMBEDDING_MODEL`, batch sizes, learning rate, epochs, ESCI→relevance mapping, split ratios.
# - Outputs: Printed summary confirming config load for later reuse.

# ============================================================================
# Step 3: Configuration and Hyperparameters
# ============================================================================

class Config:
    """Configuration class for all hyperparameters and settings"""

    # Paths (Update these based on your Kaggle dataset path)
    DATA_DIR = "/kaggle/input/amazon-esci-shopping-queries/"  # Update this!
    OUTPUT_DIR = "/kaggle/working/"

    # Model settings
    MODEL_NAME = "distilbert-base-uncased"  # Faster than BERT, good for 12hr limit
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_LENGTH = 128

    # Training hyperparameters
    BATCH_SIZE = 64  # Can increase to 64 if GPU memory allows
    INFERENCE_BATCH_SIZE = 64
    LEARNING_RATE = 2e-5
    EPOCHS = 5 # 5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0

    # Data split
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    # Evaluation
    K_VALUES = [5, 10, 20]

    # ESCI label to relevance score mapping
    ESCI_TO_RELEVANCE = {
        'E': 1.0,  # Exact
        'S': 0.7,  # Substitute
        'C': 0.3,  # Complement
        'I': 0.0   # Irrelevant
    }

    # Early stopping
    EARLY_STOPPING_PATIENCE = 2

    # Random seed
    SEED = 42

config = Config()
print("✅ Configuration loaded!")
print(f"   Model: {config.MODEL_NAME}")
print(f"   Batch Size: {config.BATCH_SIZE}")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Device: {device}")


# ### Step 4: Data Loading Functions

# - Purpose: Load parquet datasets (examples/products), filter and merge into unified frame.
# - Inputs: Parquet files in `DATA_DIR` (examples, products).
# - Key Actions: Filter `small_version==1`; merge on (`product_id`,`product_locale`); restrict to US locale; basic statistics.
# - Outputs: Merged dataframe and products dataframe returned; row counts printed.

# ============================================================================
# Step 4: Data Loading Functions
# ============================================================================

def load_dataset(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge the Amazon ESCI dataset

    Args:
        config: Configuration object

    Returns:
        Tuple of (merged_df, products_df)
    """
    print("📂 Loading dataset...")

    # Load examples
    examples_path = os.path.join(config.DATA_DIR, "shopping_queries_dataset_examples.parquet")
    df_examples = pd.read_parquet(examples_path)
    print(f"   Examples loaded: {len(df_examples):,} rows")

    # Load products
    products_path = os.path.join(config.DATA_DIR, "shopping_queries_dataset_products.parquet")
    df_products = pd.read_parquet(products_path)
    print(f"   Products loaded: {len(df_products):,} rows")

    # Filter for Task 1 (small version)
    df_examples = df_examples[df_examples["small_version"] == 1].copy()
    print(f"   Filtered for Task 1: {len(df_examples):,} rows")

    # Merge examples with products
    df_merged = pd.merge(
        df_examples,
        df_products,
        how='left',
        on=['product_id', 'product_locale']
    )
    print(f"   Merged dataset: {len(df_merged):,} rows")

    # Filter for English only (optional - remove this line for multilingual)
    df_merged = df_merged[df_merged['product_locale'] == 'us'].copy()
    print(f"   English (US) only: {len(df_merged):,} rows")

    return df_merged, df_products

def preprocess_dataset(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Preprocess the merged dataset

    Args:
        df: Merged dataframe
        config: Configuration object

    Returns:
        Preprocessed dataframe
    """
    print("\n🔧 Preprocessing dataset...")

    # Handle missing values
    df['product_title'] = df['product_title'].fillna('')
    df['product_description'] = df['product_description'].fillna('')
    df['query'] = df['query'].fillna('')

    # Combine title and description
    df['product_text'] = df['product_title'] + ' ' + df['product_description']
    print(f"   ✓ Combined title + description")

    # Convert ESCI labels to relevance scores
    df['relevance_score'] = df['esci_label'].map(config.ESCI_TO_RELEVANCE)
    print(f"   ✓ Converted ESCI labels to scores")

    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['query', 'product_id']).copy()
    print(f"   ✓ Removed {original_len - len(df):,} duplicates")

    # Keep only necessary columns
    columns_to_keep = [
        'query', 'query_id', 'product_id', 'product_text', 
        'product_title', 'product_description',
        'esci_label', 'relevance_score', 'split'
    ]
    df = df[columns_to_keep].copy()

    print(f"   ✓ Final dataset size: {len(df):,} rows")
    print(f"   ✓ Unique queries: {df['query_id'].nunique():,}")
    print(f"   ✓ Unique products: {df['product_id'].nunique():,}")

    return df

# Load and preprocess
df_merged, df_products = load_dataset(config)
df_processed = preprocess_dataset(df_merged, config)

# Display dataset statistics
print("\n📊 Dataset Statistics:")
print(df_processed['esci_label'].value_counts())
print(f"\nRelevance score distribution:")
print(df_processed['relevance_score'].describe())

# Display sample
print("\n📝 Sample rows:")
print(df_processed[['query', 'product_title', 'esci_label', 'relevance_score']].head(3))


# ### Step 5: Text Preprocessing Functions

# - Purpose: Clean and normalize text for queries and products.
# - Inputs: Raw `query`, `product_title`, `product_description` combined as `product_text`.
# - Key Actions: Expand contractions; lowercase; strip URLs/HTML; remove punctuation; tokenize; remove stopwords; lemmatize; join tokens.
# - Outputs: New columns `query_clean`, `product_text_clean`; sample preview printed.

# ============================================================================
# Step 5: Text Preprocessing Functions
# ============================================================================

class TextPreprocessor:
    """Text preprocessing pipeline for queries and products"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Expand contractions
        text = contractions.fix(text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(tokens)

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Preprocess queries and products
print("🔤 Preprocessing text data...")
df_processed['query_clean'] = preprocessor.batch_preprocess(df_processed['query'].tolist())
df_processed['product_text_clean'] = preprocessor.batch_preprocess(df_processed['product_text'].tolist())

print("✅ Text preprocessing complete!")
print(f"\nSample preprocessed query:")
print(f"Original: {df_processed.iloc[0]['query']}")
print(f"Cleaned: {df_processed.iloc[0]['query_clean']}")


# ### Step 6: Train/Val/Test Split

# - Purpose: Create train/validation/test splits without query leakage.
# - Inputs: Processed dataframe with `query_id` labels.
# - Key Actions: Split unique query IDs into train/val/test; subset rows; save splits to parquet.
# - Outputs: `train_df`, `val_df`, `test_df` saved; distribution and sizes printed.

# ============================================================================
# Step 6: Train/Val/Test Split
# ============================================================================

def create_splits(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits stratified by query_id

    Args:
        df: Processed dataframe
        config: Configuration object

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n✂️ Creating train/val/test splits...")

    # Get unique queries
    unique_queries = df['query_id'].unique()

    # Split queries (not rows) to avoid leakage
    train_queries, temp_queries = train_test_split(
        unique_queries, 
        test_size=(config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED
    )

    val_queries, test_queries = train_test_split(
        temp_queries,
        test_size=config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED
    )

    # Create splits based on query_id
    train_df = df[df['query_id'].isin(train_queries)].copy()
    val_df = df[df['query_id'].isin(val_queries)].copy()
    test_df = df[df['query_id'].isin(test_queries)].copy()

    print(f"   Train: {len(train_df):,} rows ({len(train_queries):,} queries)")
    print(f"   Val:   {len(val_df):,} rows ({len(val_queries):,} queries)")
    print(f"   Test:  {len(test_df):,} rows ({len(test_queries):,} queries)")

    # Check label distribution
    print(f"\n   Label distribution:")
    print(f"   Train: {train_df['esci_label'].value_counts(normalize=True).to_dict()}")
    print(f"   Val:   {val_df['esci_label'].value_counts(normalize=True).to_dict()}")
    print(f"   Test:  {test_df['esci_label'].value_counts(normalize=True).to_dict()}")

    return train_df, val_df, test_df

# Create splits
train_df, val_df, test_df = create_splits(df_processed, config)

# Save splits for later use
train_df.to_parquet(f"{config.OUTPUT_DIR}train_data.parquet")
val_df.to_parquet(f"{config.OUTPUT_DIR}val_data.parquet")
test_df.to_parquet(f"{config.OUTPUT_DIR}test_data.parquet")

print("✅ Splits saved!")


# ### Step 7: PyTorch Dataset Class

# - Purpose: Build PyTorch `Dataset` objects for cross-encoder training.
# - Inputs: Lists of queries, product texts, relevance scores; tokenizer; max length.
# - Key Actions: Tokenize (query, product) pairs with padding/truncation; return tensors for IDs, attention mask, label.
# - Outputs: Train/Val/Test dataset objects instantiated; sample counts printed.

# ============================================================================
# Step 7: PyTorch Dataset Class
# ============================================================================

class ProductSearchDataset(Dataset):
    """PyTorch Dataset for product search"""

    def __init__(
        self, 
        queries: List[str], 
        products: List[str], 
        labels: List[float],
        tokenizer,
        max_length: int = 128
    ):
        self.queries = queries
        self.products = products
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        product = str(self.products[idx])
        label = float(self.labels[idx])

        # Tokenize query and product together (cross-encoder style)
        encoding = self.tokenizer(
            query,
            product,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
print(f"✅ Tokenizer loaded: {config.MODEL_NAME}")

# Create datasets
train_dataset = ProductSearchDataset(
    train_df['query'].tolist(),
    train_df['product_text'].tolist(),
    train_df['relevance_score'].tolist(),
    tokenizer,
    config.MAX_LENGTH
)

val_dataset = ProductSearchDataset(
    val_df['query'].tolist(),
    val_df['product_text'].tolist(),
    val_df['relevance_score'].tolist(),
    tokenizer,
    config.MAX_LENGTH
)

test_dataset = ProductSearchDataset(
    test_df['query'].tolist(),
    test_df['product_text'].tolist(),
    test_df['relevance_score'].tolist(),
    tokenizer,
    config.MAX_LENGTH
)

print(f"✅ Datasets created:")
print(f"   Train: {len(train_dataset):,} samples")
print(f"   Val:   {len(val_dataset):,} samples")
print(f"   Test:  {len(test_dataset):,} samples")


# ### Step 8: Model Architecture

# - Purpose: Define DistilBERT-based regression model for relevance scoring.
# - Inputs: `input_ids`, `attention_mask` tensors.
# - Key Actions: Pass through encoder; take `[CLS]` embedding; apply dropout; linear layer; sigmoid activation.
# - Outputs: Scalar relevance score in [0,1]; parameter counts printed.

# ============================================================================
# Step 8: Model Architecture
# ============================================================================

class SemanticRankingModel(nn.Module):
    """
    Semantic ranking model based on DistilBERT

    Architecture:
    - DistilBERT encoder
    - Dropout layer
    - Linear regression head for relevance prediction
    """

    def __init__(self, model_name: str, dropout: float = 0.3):
        super(SemanticRankingModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

        # Initialize weights
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Dropout and regression
        x = self.dropout(cls_embedding)
        relevance_score = self.regressor(x)

        # Sigmoid to constrain to [0, 1]
        relevance_score = torch.sigmoid(relevance_score)

        return relevance_score.squeeze()

# Initialize model
model = SemanticRankingModel(config.MODEL_NAME)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model initialized!")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Model size: {total_params * 4 / 1e6:.2f} MB")


# ### Step 9: Training Setup

# - Purpose: Prepare data loaders, loss, optimizer, and LR scheduler.
# - Inputs: Dataset objects (`train_dataset`, `val_dataset`), config hyperparameters.
# - Key Actions: Instantiate `DataLoader`s; define MSE loss; set up AdamW; compute total/warmup steps; initialize linear warmup scheduler.
# - Outputs: Training setup stats printed (steps, warmup, batches).

# ============================================================================
# Step 9: Training Setup
# ============================================================================

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

# Learning rate scheduler
num_training_steps = len(train_loader) * config.EPOCHS
num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f"✅ Training setup complete!")
print(f"   Total steps: {num_training_steps:,}")
print(f"   Warmup steps: {num_warmup_steps:,}")
print(f"   Batches per epoch: {len(train_loader):,}")


# ### Step 10: Training Loop

# - Purpose: Train model across epochs with evaluation and early stopping.
# - Inputs: `train_loader`, `val_loader`, model, optimizer, scheduler, config settings.
# - Key Actions: Forward + MSE loss; backward pass; gradient clipping; scheduler stepping; validation pass; save best checkpoint; track history.
# - Outputs: `best_model.pth`, `train_history.json`, per-epoch metrics printed.

# ============================================================================
# Step 10: Training Loop
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(loader, desc="Training")
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Track loss
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    mse = np.mean((np.array(all_predictions) - np.array(all_labels)) ** 2)

    return avg_loss, mse, all_predictions, all_labels

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_mse': [],
    'learning_rates': []
}

best_val_loss = float('inf')
patience_counter = 0

print("\n🚀 Starting training...")
print("=" * 70)

for epoch in range(config.EPOCHS):
    start_time = time.time()

    print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
    print("-" * 70)

    # Train
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, scheduler, device, config
    )

    # Evaluate
    val_loss, val_mse, _, _ = evaluate(model, val_loader, criterion, device)

    # Track metrics
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))
    history['val_mse'].append(float(val_mse)) 
    history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))

    epoch_time = time.time() - start_time

    print(f"\n📊 Epoch {epoch + 1} Results:")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Val MSE: {val_mse:.4f}")
    print(f"   Time: {epoch_time:.2f}s")
    print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f"{config.OUTPUT_DIR}best_model.pth")
        print(f"   ✅ Best model saved! (Val Loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"   ⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

    # Early stopping
    if patience_counter >= config.EARLY_STOPPING_PATIENCE:
        print(f"\n⏹️ Early stopping triggered after epoch {epoch + 1}")
        break

    print("=" * 70)

print("\n✅ Training complete!")
print(f"   Best validation loss: {best_val_loss:.4f}")

# Save training history
with open(f"{config.OUTPUT_DIR}train_history.json", 'w') as f:
    json.dump(history, f, indent=2)


# ### Step 11: Training Visualization

# - Purpose: Visualize training dynamics (loss trends & learning rate schedule).
# - Inputs: `history` dictionary with tracked metrics per epoch.
# - Key Actions: Plot train vs val loss; plot LR; save figure.
# - Outputs: Displayed plots and `training_curves.png` saved to output directory.

# ============================================================================
# Step 11: Training Visualization
# ============================================================================

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning rate
axes[1].plot(history['learning_rates'], marker='o', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Schedule')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_DIR}training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Training curves saved!")


# ### Step 12: Load Best Model and Evaluate on Test Set

# - Purpose: Load best performing checkpoint and evaluate on test data.
# - Inputs: `best_model.pth`, `test_dataset`.
# - Key Actions: Restore model state; build test DataLoader; run evaluation loop computing loss & MSE.
# - Outputs: Printed test metrics (Loss, MSE, RMSE).

# ============================================================================
# Step 12: Load Best Model and Evaluate on Test Set
# ============================================================================

# Load best model
checkpoint = torch.load(f"{config.OUTPUT_DIR}best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Best model loaded from epoch {checkpoint['epoch'] + 1}")

# Create test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=config.INFERENCE_BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# Evaluate on test set
print("\n📊 Evaluating on test set...")
test_loss, test_mse, test_predictions, test_labels = evaluate(
    model, test_loader, criterion, device
)

print(f"\n📈 Test Set Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test MSE: {test_mse:.4f}")
print(f"   Test RMSE: {np.sqrt(test_mse):.4f}")


# ### Step 13: Ranking Metrics Evaluation

# - Purpose: Compute ranking-focused evaluation metrics (NDCG, Precision, Recall, F1, MAP).
# - Inputs: `test_df` grouped by `query_id`, model predictions.
# - Key Actions: Sort products per query by predicted score; derive top-K sets; aggregate metrics across queries; calculate MAP via precision at relevant ranks.
# - Outputs: Printed metric table and `evaluation_results.json` file.

# ============================================================================
# Step 13: Ranking Metrics Evaluation
# ============================================================================

def calculate_ranking_metrics(df, predictions, config):
    """
    Calculate NDCG, MAP, Precision@K, Recall@K, F1@K

    Args:
        df: Test dataframe
        predictions: Model predictions
        config: Configuration object

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import ndcg_score

    print("\n🎯 Calculating ranking metrics...")

    # Add predictions to dataframe
    df = df.copy()
    df['predicted_score'] = predictions

    metrics = {}

    # Group by query
    for k in config.K_VALUES:
        precisions = []
        recalls = []
        f1s = []
        ndcgs = []

        for query_id, group in tqdm(df.groupby('query_id'), desc=f"Metrics @{k}"):
            # Sort by predicted score
            group_sorted = group.sort_values('predicted_score', ascending=False)

            # Get top K
            top_k = group_sorted.head(k)

            # Ground truth: relevant if score > 0
            relevant_items = set(group[group['relevance_score'] > 0]['product_id'])
            retrieved_items = set(top_k['product_id'])

            # Calculate metrics
            if len(retrieved_items) > 0:
                precision = len(relevant_items & retrieved_items) / len(retrieved_items)
                precisions.append(precision)

            if len(relevant_items) > 0:
                recall = len(relevant_items & retrieved_items) / len(relevant_items)
                recalls.append(recall)

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1s.append(f1)

            # NDCG
            if len(group) > 0:
                true_relevance = group_sorted['relevance_score'].values[:k]
                pred_relevance = group_sorted['predicted_score'].values[:k]

                if len(true_relevance) > 0:
                    # Reshape for sklearn
                    ndcg = ndcg_score([true_relevance], [pred_relevance])
                    ndcgs.append(ndcg)

        metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0
        metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0
        metrics[f'f1@{k}'] = np.mean(f1s) if f1s else 0
        metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0

    # Calculate MAP
    aps = []
    for query_id, group in df.groupby('query_id'):
        group_sorted = group.sort_values('predicted_score', ascending=False)
        relevant_items = set(group[group['relevance_score'] > 0]['product_id'])

        if len(relevant_items) == 0:
            continue

        precisions_at_k = []
        num_relevant_seen = 0

        for idx, row in enumerate(group_sorted.itertuples(), 1):
            if row.product_id in relevant_items:
                num_relevant_seen += 1
                precision_at_k = num_relevant_seen / idx
                precisions_at_k.append(precision_at_k)

        if precisions_at_k:
            ap = np.mean(precisions_at_k)
            aps.append(ap)

    metrics['map'] = np.mean(aps) if aps else 0

    return metrics

# Calculate metrics
test_metrics = calculate_ranking_metrics(test_df, test_predictions, config)

print("\n" + "=" * 70)
print("📊 FINAL EVALUATION METRICS")
print("=" * 70)

print(f"\n{'Metric':<20} {'@5':<10} {'@10':<10} {'@20':<10}")
print("-" * 50)

for metric_name in ['ndcg', 'precision', 'recall', 'f1']:
    values = [
        test_metrics.get(f'{metric_name}@{k}', 0) 
        for k in config.K_VALUES
    ]
    print(f"{metric_name.upper():<20} {values[0]:<10.4f} {values[1]:<10.4f} {values[2]:<10.4f}")

print(f"\nMAP: {test_metrics['map']:.4f}")
print("=" * 70)

# Save metrics
with open(f"{config.OUTPUT_DIR}evaluation_results.json", 'w') as f:
    json.dump(test_metrics, f, indent=2)

print("\n✅ Evaluation complete! Results saved.")


# ### Step 14: Generate Product Embeddings for Fast Retrieval

# - Purpose: Generate dense product embeddings for fast semantic retrieval.
# - Inputs: Unique US-locale products (`product_title` + `product_description`).
# - Key Actions: Concatenate text fields; batch encode with SentenceTransformer; stack arrays; save embeddings & metadata.
# - Outputs: `product_embeddings.npy`, `product_metadata.parquet`, shape stats printed.

# ============================================================================
# Step 14: Generate Product Embeddings for Fast Retrieval
# ============================================================================

print("\n🔍 Generating product embeddings for deployment...")

# Get unique products from full dataset
unique_products = df_products[df_products['product_locale'] == 'us'].copy()
unique_products['product_text'] = (
    unique_products['product_title'].fillna('') + ' ' + 
    unique_products['product_description'].fillna('')
)

print(f"   Total unique products: {len(unique_products):,}")

# Load sentence transformer for embedding generation
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
embedding_model = embedding_model.to(device)

# Generate embeddings in batches
product_embeddings = []
batch_size = 128

for i in tqdm(range(0, len(unique_products), batch_size), desc="Encoding products"):
    batch = unique_products.iloc[i:i+batch_size]['product_text'].tolist()
    embeddings = embedding_model.encode(
        batch,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    product_embeddings.append(embeddings)

product_embeddings = np.vstack(product_embeddings)

print(f"✅ Product embeddings generated!")
print(f"   Shape: {product_embeddings.shape}")

# Save embeddings and product metadata
np.save(f"{config.OUTPUT_DIR}product_embeddings.npy", product_embeddings)
unique_products[['product_id', 'product_title', 'product_description', 'product_text']].to_parquet(
    f"{config.OUTPUT_DIR}product_metadata.parquet"
)

print("✅ Embeddings and metadata saved!")


# ### Step 15: Create Results Visualization

# - Purpose: Produce comprehensive visual summary of training & evaluation results.
# - Inputs: `history`, `test_metrics`, `test_df`, `test_predictions`.
# - Key Actions: Build multi-panel figure (loss curves, metrics bar, NDCG vs K, scatter, label distribution, metrics table); annotate cells; save figure.
# - Outputs: Displayed figure and `comprehensive_results.png`.

# ============================================================================
# Step 15: Create Results Visualization
# ============================================================================

# Create comprehensive results figure
fig = plt.figure(figsize=(20, 12))

# 1. Training curves (top left)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history['train_loss'], label='Train Loss', marker='o')
ax1.plot(history['val_loss'], label='Val Loss', marker='s')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Metrics bar chart (top middle)
ax2 = plt.subplot(2, 3, 2)
metric_names = ['NDCG', 'Precision', 'Recall', 'F1']
k10_values = [
    test_metrics.get('ndcg@10', 0),
    test_metrics.get('precision@10', 0),
    test_metrics.get('recall@10', 0),
    test_metrics.get('f1@10', 0)
]
bars = ax2.bar(metric_names, k10_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Metrics @10', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
for bar, value in zip(bars, k10_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.3f}', ha='center', va='bottom', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. NDCG comparison across K values (top right)
ax3 = plt.subplot(2, 3, 3)
k_vals = config.K_VALUES
ndcg_vals = [test_metrics.get(f'ndcg@{k}', 0) for k in k_vals]
ax3.plot(k_vals, ndcg_vals, marker='o', linewidth=2, markersize=10, color='#1f77b4')
ax3.set_xlabel('K', fontsize=12)
ax3.set_ylabel('NDCG@K', fontsize=12)
ax3.set_title('NDCG Across Different K Values', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(k_vals)

# 4. Prediction vs True scores scatter (bottom left)
ax4 = plt.subplot(2, 3, 4)
sample_size = min(5000, len(test_labels))
sample_indices = np.random.choice(len(test_labels), sample_size, replace=False)
ax4.scatter(
    np.array(test_labels)[sample_indices],
    np.array(test_predictions)[sample_indices],
    alpha=0.3,
    s=10
)
ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('True Relevance Score', fontsize=12)
ax4.set_ylabel('Predicted Relevance Score', fontsize=12)
ax4.set_title('Prediction vs Ground Truth', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. ESCI label distribution (bottom middle)
ax5 = plt.subplot(2, 3, 5)
label_counts = test_df['esci_label'].value_counts()
colors_esci = {'E': '#2ca02c', 'S': '#ff7f0e', 'C': '#d62728', 'I': '#9467bd'}
bars = ax5.bar(
    label_counts.index, 
    label_counts.values,
    color=[colors_esci.get(label, '#1f77b4') for label in label_counts.index]
)
ax5.set_xlabel('ESCI Label', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.set_title('Test Set Label Distribution', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Metrics comparison table (bottom right)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

table_data = []
table_data.append(['Metric', '@5', '@10', '@20'])
for metric in ['ndcg', 'precision', 'recall', 'f1']:
    row = [metric.upper()]
    for k in config.K_VALUES:
        val = test_metrics.get(f'{metric}@{k}', 0)
        row.append(f'{val:.4f}')
    table_data.append(row)
table_data.append(['MAP', f"{test_metrics['map']:.4f}", '-', '-'])

table = ax6.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.3, 0.2, 0.2, 0.2]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax6.set_title('Evaluation Metrics Summary', fontsize=14, fontweight='bold', pad=20)

plt.suptitle(
    'Semantic Product Search - Training Results & Evaluation',
    fontsize=16,
    fontweight='bold',
    y=0.995
)

plt.tight_layout()
plt.savefig(f"{config.OUTPUT_DIR}comprehensive_results.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comprehensive results visualization saved!")


# ### Step 16: Save Final Package for Local Deployment

# - Purpose: Package final artifacts and provide deployment instructions.
# - Inputs: Generated model, embeddings, metrics, dataset splits.
# - Key Actions: Enumerate files with sizes; build summary JSON (config, stats, performance); print next-step checklist.
# - Outputs: `model_summary.json` plus confirmation list of downloadable files.

# ============================================================================
# Step 16: Save Final Package for Local Deployment
# ============================================================================

print("\n📦 Preparing files for local deployment...")

# Files to download:
files_to_download = [
    "best_model.pth",
    "product_embeddings.npy",
    "product_metadata.parquet",
    "train_history.json",
    "evaluation_results.json",
    "training_curves.png",
    "comprehensive_results.png",
    "train_data.parquet",
    "val_data.parquet",
    "test_data.parquet"
]

print("\n📥 Download these files from Kaggle output:")
for file in files_to_download:
    file_path = os.path.join(config.OUTPUT_DIR, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   ✅ {file} ({size:.2f} MB)")
    else:
        print(f"   ❌ {file} (not found)")

# Create a summary report
summary = {
    "model_name": config.MODEL_NAME,
    "embedding_model": config.EMBEDDING_MODEL,
    "training_config": {
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "max_length": config.MAX_LENGTH
    },
    "dataset_stats": {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "unique_queries": df_processed['query_id'].nunique(),
        "unique_products": unique_products.shape[0]
    },
    "best_model": {
        "epoch": checkpoint['epoch'] + 1,
        "val_loss": checkpoint['val_loss']
    },
    "test_performance": test_metrics,
    "training_time": {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
}

with open(f"{config.OUTPUT_DIR}model_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Model summary saved!")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print("\n📋 Next Steps:")
print("   1. Download all files from Kaggle output section")
print("   2. Place files in local 'models/' directory")
print("   3. Run Streamlit app locally: streamlit run app.py")
print("\n" + "=" * 70)

