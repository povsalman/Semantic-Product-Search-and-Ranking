"""
Data Loading Utilities for Semantic Product Search
===================================================

This module handles loading and preprocessing of the Amazon ESCI dataset
for semantic product search and ranking.

Author: [Your Name]
Course: Generative AI - Fall 2025
"""

import os
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader for Amazon ESCI Shopping Queries Dataset
    
    Handles loading, merging, and basic preprocessing of examples and products.
    """
    
    def __init__(self, data_dir: str = "data/raw/"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing the parquet files
        """
        self.data_dir = data_dir
        self.examples_file = "shopping_queries_dataset_examples.parquet"
        self.products_file = "shopping_queries_dataset_products.parquet"
        
    def load_examples(self) -> pd.DataFrame:
        """
        Load examples (query-product pairs) from parquet file
        
        Returns:
            DataFrame containing examples
        """
        path = os.path.join(self.data_dir, self.examples_file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Examples file not found: {path}\n"
                f"Please download from: "
                f"https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset"
            )
        
        df = pd.read_parquet(path)
        print(f"✓ Loaded {len(df):,} examples")
        
        return df
    
    def load_products(self) -> pd.DataFrame:
        """
        Load product metadata from parquet file
        
        Returns:
            DataFrame containing product information
        """
        path = os.path.join(self.data_dir, self.products_file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Products file not found: {path}\n"
                f"Please download from: "
                f"https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset"
            )
        
        df = pd.read_parquet(path)
        print(f"✓ Loaded {len(df):,} products")
        
        return df
    
    def merge_data(
        self, 
        examples_df: pd.DataFrame, 
        products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge examples with product information
        
        Args:
            examples_df: DataFrame with examples
            products_df: DataFrame with products
        
        Returns:
            Merged DataFrame
        """
        merged = pd.merge(
            examples_df,
            products_df,
            how='left',
            on=['product_id', 'product_locale']
        )
        
        print(f"✓ Merged dataset: {len(merged):,} rows")
        
        return merged
    
    def filter_for_task1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset for Task 1 (ranking task - small version)
        
        Args:
            df: Merged DataFrame
        
        Returns:
            Filtered DataFrame
        """
        df_filtered = df[df['small_version'] == 1].copy()
        print(f"✓ Filtered for Task 1: {len(df_filtered):,} rows")
        
        return df_filtered
    
    def filter_by_locale(
        self, 
        df: pd.DataFrame, 
        locale: str = 'us'
    ) -> pd.DataFrame:
        """
        Filter dataset by product locale
        
        Args:
            df: DataFrame to filter
            locale: Product locale ('us', 'es', 'jp')
        
        Returns:
            Filtered DataFrame
        """
        df_filtered = df[df['product_locale'] == locale].copy()
        print(f"✓ Filtered for locale '{locale}': {len(df_filtered):,} rows")
        
        return df_filtered
    
    def prepare_dataset(
        self,
        locale: str = 'us',
        task: int = 1
    ) -> pd.DataFrame:
        """
        Complete pipeline: load, merge, and filter dataset
        
        Args:
            locale: Product locale to filter
            task: Task number (1 for ranking)
        
        Returns:
            Prepared DataFrame
        """
        print("\n📂 Loading and preparing dataset...")
        print("=" * 60)
        
        # Load data
        examples = self.load_examples()
        products = self.load_products()
        
        # Merge
        merged = self.merge_data(examples, products)
        
        # Filter for task
        if task == 1:
            merged = self.filter_for_task1(merged)
        
        # Filter by locale
        merged = self.filter_by_locale(merged, locale)
        
        # Basic preprocessing
        merged = self._basic_preprocessing(merged)
        
        print("=" * 60)
        print(f"✅ Dataset ready: {len(merged):,} rows\n")
        
        return merged
    
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic preprocessing steps
        
        Args:
            df: DataFrame to preprocess
        
        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        df['product_title'] = df['product_title'].fillna('')
        df['product_description'] = df['product_description'].fillna('')
        df['query'] = df['query'].fillna('')
        
        # Combine title and description
        df['product_text'] = (
            df['product_title'] + ' ' + df['product_description']
        )
        
        # Convert ESCI labels to numeric relevance scores
        esci_to_relevance = {
            'E': 1.0,  # Exact
            'S': 0.7,  # Substitute
            'C': 0.3,  # Complement
            'I': 0.0   # Irrelevant
        }
        df['relevance_score'] = df['esci_label'].map(esci_to_relevance)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['query', 'product_id']).copy()
        
        print(f"✓ Basic preprocessing complete")
        
        return df


def load_processed_data(data_dir: str = "data/processed/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split train/val/test datasets
    
    Args:
        data_dir: Directory containing processed parquet files
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_path = os.path.join(data_dir, "train_data.parquet")
    val_path = os.path.join(data_dir, "val_data.parquet")
    test_path = os.path.join(data_dir, "test_data.parquet")
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            "Processed data files not found. Please run Kaggle training first."
        )
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"✓ Train: {len(train_df):,} rows")
    print(f"✓ Val: {len(val_df):,} rows")
    print(f"✓ Test: {len(test_df):,} rows")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.prepare_dataset(locale='us', task=1)
    
    print("\n📊 Dataset Statistics:")
    print(f"Unique queries: {df['query_id'].nunique():,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print("\nESCI Label Distribution:")
    print(df['esci_label'].value_counts())