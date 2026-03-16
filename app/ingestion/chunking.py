def split_by_tokens(text, tokenizer, max_tokens=1024, overlap=100):
    """
    按 token 数分割文本，并保留一定的重叠。
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks
