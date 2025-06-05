import sentencepiece as spm
import logging
import os

def load_tokenizer(tokenizer_path):
    """
    Load a SentencePiece tokenizer from the specified path.
    
    Args:
        tokenizer_path (str): Path to the SentencePiece model file (e.g., story_tokenizer.model).
    
    Returns:
        spm.SentencePieceProcessor: Loaded tokenizer instance.
    
    Raises:
        FileNotFoundError: If the tokenizer file does not exist.
        RuntimeError: If loading the tokenizer fails.
    """
    logging.debug(f"Attempting to load tokenizer from {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer file not found: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found")
    
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        logging.info(f"Successfully loaded tokenizer from {tokenizer_path} with vocab size: {tokenizer.get_piece_size()}")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        raise RuntimeError(f"Failed to load tokenizer: {e}")