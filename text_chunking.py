def split_text(text, chunk_size=100, overlap=50):
    """
    Splits a large text into smaller equal-sized chunks with a specified overlap.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The size of each chunk. Default is 100 characters.
        overlap (int): The number of overlapping characters between chunks. Default is 50 characters.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0

    # Ensure chunk size is larger than overlap
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    while start < len(text):
        # Extract a chunk
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move the start position forward by chunk_size - overlap
        start += chunk_size - overlap

    return chunks
