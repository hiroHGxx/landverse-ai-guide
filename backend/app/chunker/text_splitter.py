import re
from typing import List


def split_text_into_chunks(text: str, max_chunk_size: int = 1000, overlap_size: int = 100) -> List[str]:
    """
    Split long text into semantically meaningful chunks with natural breakpoints.
    
    Args:
        text: The long text string to be split
        max_chunk_size: Maximum number of characters per chunk (default: 1000)
        overlap_size: Number of characters to overlap between chunks (default: 100)
        
    Returns:
        List of text chunks split at natural breakpoints
    """
    if not text or not text.strip():
        return []
    
    if len(text) <= max_chunk_size:
        return [text.strip()]
    
    chunks = []
    
    # First, split by paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed max_chunk_size
        if len(current_chunk) + len(paragraph) + 1 > max_chunk_size:
            if current_chunk:
                # Process current chunk before starting new one
                chunks.extend(_split_large_text(current_chunk, max_chunk_size, overlap_size))
                current_chunk = ""
            
            # If single paragraph is too large, split it further
            if len(paragraph) > max_chunk_size:
                chunks.extend(_split_large_text(paragraph, max_chunk_size, overlap_size))
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.extend(_split_large_text(current_chunk, max_chunk_size, overlap_size))
    
    # Remove empty chunks and strip whitespace
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_large_text(text: str, max_chunk_size: int, overlap_size: int) -> List[str]:
    """
    Split large text using sentence and clause boundaries.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum chunk size
        overlap_size: Overlap size between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    
    # Split by sentences first
    sentences = _split_into_sentences(text)
    
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chunk_size
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap for next chunk
                if overlap_size > 0:
                    overlap_text = _get_overlap_text(current_chunk, overlap_size)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
            else:
                # Single sentence is too large, split by clauses/phrases
                if len(sentence) > max_chunk_size:
                    chunks.extend(_split_by_clauses(sentence, max_chunk_size))
                else:
                    current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation markers.
    
    Args:
        text: Text to split into sentences
        
    Returns:
        List of sentences
    """
    # Split by sentence-ending punctuation (Japanese and English)
    sentence_pattern = r'[.!?。！？]+\s*'
    sentences = re.split(sentence_pattern, text)
    
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def _split_by_clauses(text: str, max_chunk_size: int) -> List[str]:
    """
    Split text by clauses and phrases when sentences are too long.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by commas, semicolons, and conjunctions
    clause_pattern = r'[,;、]|\s+(?:and|or|but|however|また|そして|しかし|ただし)\s+'
    clauses = re.split(clause_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
            
        if len(current_chunk) + len(clause) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = clause
            else:
                # If single clause is still too large, split by character limit
                chunks.extend(_split_by_character_limit(clause, max_chunk_size))
        else:
            if current_chunk:
                current_chunk += ", " + clause
            else:
                current_chunk = clause
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _split_by_character_limit(text: str, max_chunk_size: int) -> List[str]:
    """
    Split text by character limit as a last resort, trying to break at word boundaries.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of text chunks
    """
    chunks = []
    words = text.split()
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                # Single word is too large, split by character
                if len(word) > max_chunk_size:
                    for i in range(0, len(word), max_chunk_size):
                        chunks.append(word[i:i + max_chunk_size])
                else:
                    current_chunk = word
        else:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def _get_overlap_text(text: str, overlap_size: int) -> str:
    """
    Get the last part of text for overlap with next chunk.
    
    Args:
        text: Source text
        overlap_size: Number of characters for overlap
        
    Returns:
        Overlap text
    """
    if len(text) <= overlap_size:
        return text
    
    # Try to find a good breakpoint for overlap (sentence or word boundary)
    overlap_text = text[-overlap_size:]
    
    # Find the first sentence boundary in the overlap
    sentence_match = re.search(r'[.!?。！？]\s+', overlap_text)
    if sentence_match:
        return overlap_text[sentence_match.end():]
    
    # Find the first word boundary
    word_match = re.search(r'\s+', overlap_text)
    if word_match:
        return overlap_text[word_match.end():]
    
    return overlap_text