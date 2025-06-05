import torch
import torch.nn.functional as F
import logging
from dataloader.data_utils import clean_genre_label

def generate_story(model, tokenizer, query, label2id=None, genre=None, max_length=256, temperature=0.9, pad_idx=0, eos_idx=3, top_k=50, top_p=0.9):
    """Generate a single story based on a query and optional genre."""
    model.eval()
    device = next(model.parameters()).device
    tokenizer_vocab_size = tokenizer.get_piece_size()
    model_vocab_size = model.lm_head.out_features
    if tokenizer_vocab_size != model_vocab_size:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) differs from model vocab size ({model_vocab_size})")
    
    tokens = tokenizer.encode(query, out_type=int)
    if genre and label2id:
        genre = clean_genre_label(genre)
        genre_token_id = label2id.get(genre, tokenizer.piece_to_id("<Unknown>"))
        tokens = [genre_token_id] + tokens
    if len(tokens) >= max_length:
        tokens = tokens[:max_length-1]
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = input_ids.clone()

    def create_causal_mask(seq_len, nhead):
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        return mask.unsqueeze(0).unsqueeze(1).expand(1, nhead, seq_len, seq_len).reshape(nhead, seq_len, seq_len)

    def top_p_filtering(logits, top_p):
        batch_size, vocab_size = logits.shape
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        mask = torch.ones_like(logits, dtype=torch.bool)
        for b in range(batch_size):
            mask[b, sorted_indices[b, sorted_indices_to_remove[b]]] = False
        logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        return logits

    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            seq_len = generated.shape[1]
            tgt_mask = create_causal_mask(seq_len, nhead=model.nhead)
            tgt_key_padding_mask = (generated == pad_idx).to(device)
            logits = model(generated, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.warning("NaN or inf detected in logits, clamping values")
                logits = torch.clamp(logits, min=-1e9, max=1e9)
            next_token_logits = logits[:, -1, :] / temperature
            if next_token_logits.shape[1] != model_vocab_size:
                logging.error(f"Logits shape mismatch: expected [*, {model_vocab_size}], got {next_token_logits.shape}")
                break
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            if torch.isnan(next_token_probs).any() or torch.isinf(next_token_probs).any():
                logging.warning("NaN or inf detected in probabilities, using uniform distribution")
                next_token_probs = torch.ones_like(next_token_probs) / model_vocab_size
            if top_p is not None:
                next_token_logits = top_p_filtering(next_token_logits, top_p)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token)
            else:
                next_token = torch.multinomial(next_token_probs, num_samples=1)
            next_token = torch.clamp(next_token, min=0, max=model_vocab_size-1)
            next_token = next_token.detach().view(-1, 1).to(device)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in [pad_idx, eos_idx]:
                break
    tokens = generated.squeeze(0).cpu().tolist()
    tokens = [t for t in tokens if 0 <= t < tokenizer_vocab_size]
    try:
        generated_text = tokenizer.decode(tokens)
    except Exception as e:
        logging.error(f"Failed to decode tokens: {e}")
        generated_text = "<Decoding failed>"
    genre_token = f"<{genre}>" if genre else None
    if genre_token and generated_text.startswith(genre_token):
        generated_text = generated_text[len(genre_token):].strip()
    logging.info(f"Generated story for query '{query[:50]}...' (genre: {genre or 'None'}): {generated_text[:100]}...")
    return generated_text

def generate_multiple_stories(model, tokenizer, queries, label2id=None, genres=None, max_length=256, temperature=0.9, pad_idx=0, eos_idx=3, top_k=50, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    tokenizer_vocab_size = tokenizer.get_piece_size()
    model_vocab_size = model.lm_head.out_features
    if tokenizer_vocab_size != model_vocab_size:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) differs from model vocab size ({model_vocab_size})")
    if genres is None:
        genres = [None] * len(queries)
    elif len(genres) != len(queries):
        raise ValueError("Number of genres must match number of queries")
    batch_size = len(queries)
    logging.debug(f"Batch size: {batch_size}, model.nhead: {model.nhead}")
    batch_tokens = []
    batch_genres = []
    for query, genre in zip(queries, genres):
        genre = clean_genre_label(genre) if genre else None
        tokens = tokenizer.encode(query, out_type=int)
        if genre and label2id:
            genre_token_id = label2id.get(genre, tokenizer.piece_to_id("<Unknown>"))
            tokens = [genre_token_id] + tokens
        if len(tokens) >= max_length:
            tokens = tokens[:max_length-1]
        batch_tokens.append(tokens)
        batch_genres.append(genre)
    max_seq_len = min(max(max(len(t) for t in batch_tokens), 1), max_length)
    input_ids = torch.full((batch_size, max_seq_len), pad_idx, dtype=torch.long, device=device)
    for i, tokens in enumerate(batch_tokens):
        input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    generated = input_ids.clone()

    def create_causal_mask(seq_len, nhead, batch_size):
        logging.debug(f"Creating causal mask: seq_len={seq_len}, nhead={nhead}, batch_size={batch_size}")
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nhead, seq_len, seq_len)
        reshaped_mask = mask.reshape(batch_size * nhead, seq_len, seq_len)
        logging.debug(f"Causal mask shape: {reshaped_mask.shape}")
        return reshaped_mask

    def top_p_filtering(logits, top_p):
        batch_size, vocab_size = logits.shape
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        mask = torch.ones_like(logits, dtype=torch.bool)
        for b in range(batch_size):
            mask[b, sorted_indices[b, sorted_indices_to_remove[b]]] = False
        logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        return logits

    with torch.no_grad():
        for _ in range(max_length - max_seq_len):
            seq_len = generated.shape[1]
            tgt_mask = create_causal_mask(seq_len, nhead=model.nhead, batch_size=batch_size)
            tgt_key_padding_mask = (generated == pad_idx)
            logging.debug(f"Generated shape: {generated.shape}")
            logging.debug(f"tgt_mask shape: {tgt_mask.shape}")
            logging.debug(f"tgt_key_padding_mask shape: {tgt_key_padding_mask.shape}")
            logits = model(generated, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.warning("NaN or inf detected in logits, clamping values")
                logits = torch.clamp(logits, min=-1e9, max=1e9)
            next_token_logits = logits[:, -1, :] / temperature
            if next_token_logits.shape[1] != model_vocab_size:
                logging.error(f"Logits shape mismatch: expected [*, {model_vocab_size}], got {next_token_logits.shape}")
                break
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            if torch.isnan(next_token_probs).any() or torch.isinf(next_token_probs).any():
                logging.warning("NaN or inf detected in probabilities, using uniform distribution")
                next_token_probs = torch.ones_like(next_token_probs) / model_vocab_size
            if top_p is not None:
                next_token_logits = top_p_filtering(next_token_logits, top_p)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token)
            else:
                next_token = torch.multinomial(next_token_probs, num_samples=1)
            next_token = torch.clamp(next_token, min=0, max=model_vocab_size-1)
            next_token = next_token.detach()
            generated = torch.cat([generated, next_token], dim=1)
            done = (next_token.squeeze(-1) == pad_idx) | (next_token.squeeze(-1) == eos_idx)
            active = torch.ones(batch_size, dtype=torch.bool, device=device) & ~done
            if not active.any():
                break
    results = []
    for i, (query, genre) in enumerate(zip(queries, batch_genres)):
        tokens = generated[i].cpu().tolist()
        tokens = [t for t in tokens if 0 <= t < tokenizer_vocab_size]
        try:
            text = tokenizer.decode(tokens)
        except Exception as e:
            logging.error(f"Failed to decode tokens for query {i+1}: {e}")
            text = "<Decoding failed>"
        genre_token = f"<{genre}>" if genre else None
        if genre_token and text.startswith(genre_token):
            text = text[len(genre_token):].strip()
        results.append((query, genre, text))
        logging.info(f"Generated story {i+1} (genre: {genre or 'None'}): {text[:100]}...")
    return results