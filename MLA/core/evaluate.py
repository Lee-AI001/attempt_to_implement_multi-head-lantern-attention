
import torch
import math
import logging
from tqdm import tqdm
from core.generate import generate_multiple_stories

def evaluate(model, data_loader, criterion, device, vocab_size, pad_idx=0):
    model.eval()
    total_loss = 0
    total_batches = 0
    model.to(device)
    if len(data_loader) == 0:
        logging.warning("Validation data loader is empty!")
        return {"loss": float("inf"), "perplexity": float("inf")}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            logits = model(input_ids, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            if pad_idx is not None:
                mask = target_ids != pad_idx
                loss = (loss * mask.view(-1)).sum() / mask.sum() if mask.sum() > 0 else loss
            total_loss += loss.item()
            total_batches += 1
    avg_loss = total_loss / total_batches
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
    return {"loss": avg_loss, "perplexity": perplexity}

def compute_rouge_with_subset(model, tokenizer, dev_data, label2id, num_samples=50, max_length=256, temperature=0.9, pad_idx=0, eos_idx=3, top_k=50, top_p=0.9):
    """Compute ROUGE scores using a random subset of dev_data."""
    from rouge_score import rouge_scorer
    import random

    # Sample a subset of dev_data
    subset = random.sample(dev_data, min(num_samples, len(dev_data)))
    queries = [sample["text"] for sample in subset]  # Assuming "text" is the query field
    reference_texts = [sample["text"] for sample in subset]  # Assuming "text" is the reference

    # Generate stories
    results = generate_multiple_stories(
        model, tokenizer, queries, label2id=label2id, max_length=max_length,
        temperature=temperature, pad_idx=pad_idx, eos_idx=eos_idx, top_k=top_k, top_p=top_p
    )
    generated_stories = [result[2] for result in results]  # Assuming result[2] is the story text

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []
    for gen, ref in zip(generated_stories, reference_texts):
        score = scorer.score(ref, gen)
        scores.append((score['rouge1'].fmeasure + score['rougeL'].fmeasure) / 2)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    logging.info(f"Computed ROUGE score on {len(subset)} samples: {avg_score:.4f}")
    return avg_score