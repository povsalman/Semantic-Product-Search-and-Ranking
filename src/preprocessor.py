"""
Text Preprocessing Module for Semantic Product Search
======================================================

Handles all text preprocessing including:
- Contraction expansion
- Lowercasing
- Special character removal
- Stop word removal
- Lemmatization

Author: [Your Name]
Course: Generative AI - Fall 2025
"""

import re
import string
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """
    Text preprocessing pipeline for product search queries and descriptions
    
    Applies standard NLP preprocessing techniques to clean and normalize text.
    """
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True,
        min_token_length: int = 2
    ):
        """
        Initialize text preprocessor
        
        Args:
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to apply lemmatization
            lowercase: Whether to convert to lowercase
            min_token_length: Minimum token length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        
        # Initialize NLTK components
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions (e.g., "don't" -> "do not")
        
        Args:
            text: Input text
        
        Returns:
            Text with expanded contractions
        """
        return contractions.fix(text)
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
        
        Returns:
            Text without URLs
        """
        return re.sub(
            r'http\S+|www\S+|https\S+', 
            '', 
            text, 
            flags=re.MULTILINE
        )
    
    def remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text
        
        Args:
            text: Input text
        
        Returns:
            Text without HTML tags
        """
        return re.sub(r'<.*?>', '', text)
    
    def remove_special_chars(self, text: str) -> str:
        """
        Remove special characters, keeping only alphanumeric and spaces
        
        Args:
            text: Input text
        
        Returns:
            Text with only alphanumeric characters and spaces
        """
        # Keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered list of tokens
        """
        return [
            token for token in tokens 
            if token.lower() not in self.stop_words
        ]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens: List of tokens
        
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered tokens
        """
        return [
            token for token in tokens 
            if len(token) >= self.min_token_length
        ]
    
    def preprocess(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline
        
        Pipeline:
        1. Handle non-string input
        2. Expand contractions
        3. Remove URLs and HTML
        4. Lowercase
        5. Remove special characters
        6. Tokenize
        7. Remove stop words
        8. Lemmatize
        9. Filter by length
        10. Join back to string
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        # Handle non-string input
        if not isinstance(text, str):
            return ""
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove URLs and HTML
        text = self.remove_urls(text)
        text = self.remove_html(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters
        text = self.remove_special_chars(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Filter by length
        tokens = self.filter_by_length(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def batch_preprocess(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of texts to preprocess
            show_progress: Whether to show progress bar
        
        Returns:
            List of preprocessed texts
        """
        if show_progress:
            from tqdm import tqdm
            return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
        else:
            return [self.preprocess(text) for text in texts]


class MinimalPreprocessor:
    """
    Minimal preprocessing for use with pretrained models like BERT
    
    Only applies basic cleaning without removing stop words or lemmatization,
    as these models are trained on natural language.
    """
    
    def __init__(self, lowercase: bool = True):
        """
        Initialize minimal preprocessor
        
        Args:
            lowercase: Whether to convert to lowercase
        """
        self.lowercase = lowercase
    
    def preprocess(self, text: str) -> str:
        """
        Apply minimal preprocessing
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML
        text = re.sub(r'<.*?>', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def batch_preprocess(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of texts to preprocess
            show_progress: Whether to show progress bar
        
        Returns:
            List of preprocessed texts
        """
        if show_progress:
            from tqdm import tqdm
            return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
        else:
            return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Test the preprocessor
    test_texts = [
        "I'm looking for wireless Bluetooth headphones with noise cancellation!",
        "Best gaming laptop 2024 - RTX 4090, 32GB RAM",
        "Women's running shoes <size 8> @ $50-100",
        "Don't you think this product's awesome? Check www.example.com!"
    ]
    
    print("=" * 70)
    print("Testing Full Preprocessor")
    print("=" * 70)
    
    preprocessor = TextPreprocessor()
    for text in test_texts:
        cleaned = preprocessor.preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
    
    print("\n" + "=" * 70)
    print("Testing Minimal Preprocessor (for BERT)")
    print("=" * 70)
    
    minimal = MinimalPreprocessor()
    for text in test_texts:
        cleaned = minimal.preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")