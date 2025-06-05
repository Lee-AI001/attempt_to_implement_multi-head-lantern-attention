import re
import os
import json
import random
import logging
from pathlib import Path

def clean_genre_label(genre):
    """Clean genre labels to keep alphanumeric characters and replace spaces with underscores."""
    if not isinstance(genre, str):
        return None
    cleaned = genre.strip().replace(" ", "_")
    cleaned = re.sub(r'[^\w_]', '', cleaned)
    return cleaned if cleaned else None

def extract_genre_labels(data):
    """Extract unique genre labels from the dataset."""
    genre_set = set()
    for sample in data:
        if "labels" in sample and isinstance(sample["labels"], list):
            for genre in sample["labels"]:
                cleaned_genre = clean_genre_label(genre)
                if cleaned_genre:
                    genre_set.add(cleaned_genre)
    genres = sorted(list(genre_set))
    if not genres:
        logging.warning("No valid genre labels found in dataset")
        print("⚠️ No valid genre labels found")
    else:
        logging.info(f"Extracted {len(genres)} unique genres: {genres}")
        print(f"✅ Extracted {len(genres)} unique genres")
    return genres

def clean_data(sample):
    """Clean a single JSONL sample by validating and processing text."""
    try:
        if not isinstance(sample, dict) or not sample.get("body"):
            logging.warning("Invalid sample: missing or empty 'body' field")
            return None
        text = sample["body"].strip()
        if not text or len(text) < 5:
            logging.warning(f"Sample too short: {text[:50]}...")
            return None
        cleaned_text = " ".join(text.split())
        cleaned_sample = {"text": cleaned_text}
        if "type" in sample and isinstance(sample["type"], list):
            cleaned_sample["labels"] = sample["type"]
        return cleaned_sample
    except Exception as e:
        logging.warning(f"Failed to process sample: {e}")
        return None

def load_and_clean_data(data_files, max_size_mb=0):
    """Load and clean data from multiple JSONL files with size limit per file."""
    data = []
    total_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb > 0 else float("inf")
    skipped_samples = 0

    for file_path in data_files:
        file_path = Path(file_path)
        if not file_path.exists():
            logging.error(f"Data file {file_path} does not exist")
            raise FileNotFoundError(f"Data file {file_path} does not exist")
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logging.info(f"Loading data from {file_path}, file size: {file_size_mb:.2f} MB")
        if file_size_mb == 0:
            logging.error(f"Data file {file_path} is empty")
            raise ValueError(f"Data file {file_path} is empty")

        file_data = []
        file_size = 0
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if file_size > max_size_bytes:
                        logging.warning(f"Reached max size limit ({max_size_mb} MB) for {file_path}")
                        break
                    line = line.strip()
                    if not line:
                        skipped_samples += 1
                        continue
                    try:
                        sample = json.loads(line)
                        cleaned_sample = clean_data(sample)
                        if cleaned_sample:
                            file_data.append(cleaned_sample)
                            file_size += len(line.encode("utf-8"))
                        else:
                            skipped_samples += 1
                            logging.warning(f"Sample skipped in {file_path}: {line[:50]}...")
                    except json.JSONDecodeError as e:
                        skipped_samples += 1
                        logging.warning(f"Invalid JSONL line in {file_path}: {line[:50]}... Error: {e}")
            logging.info(f"Loaded {len(file_data)} samples from {file_path}, skipped {skipped_samples} samples")
            data.extend(file_data)
            total_size += file_size
        except Exception as e:
            logging.error(f"Failed to load data from {file_path}: {e}")
            raise

    if not data:
        logging.error("No valid data loaded from any dataset")
        raise ValueError("No valid data loaded from any dataset")
    logging.info(f"Total loaded: {len(data)} samples, total size: {total_size / (1024 * 1024):.2f} MB")
    return data

def split_data(data, split_ratio=0.8):
    """Split data into train and dev sets."""
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    dev_data = data[split_index:]
    logging.info(f"Train set: {len(train_data)} samples, Dev set: {len(dev_data)} samples")
    print(f"✅ Train set: {len(train_data)} samples")
    print(f"✅ Dev set: {len(dev_data)} samples")
    return train_data, dev_data

def write_train_texts(train_data, output_path):
    """Write training texts to a file for tokenizer training."""
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8", errors="replace") as f:
        for sample in train_data:
            if isinstance(sample["text"], str) and sample["text"].strip():
                f.write(sample["text"].strip() + "\n")
    logging.info(f"Wrote training texts to {output_path}")
    print(f"✅ Wrote training texts to {output_path}")

def save_model_config(
    project_dir, base_dir, project_name, data_files, vocab_size, num_layers,
    d_model, nhead, dropout, embed_dropout, dim_feedforward, max_len, padding_value,
    sliding_step, max_chunks, learning_rate, weight_decay, batch_size, start_epoch,
    epochs, max_grad_norm, patience, use_opt, scheduler_type, warmup_epochs,
    split_ratio, max_gen_len, temperature, pad_idx, eos_idx, save_dis, device,
    label_smoothing, grad_accum_steps, test_size, rouge_sample_size, mixed_precision,
    plateau_factor, plateau_patience, use_optuna, optuna_n_trials,
    num_genres, d_c, d_prime_c, d_R_h
):
    """Save model configuration to a JSON file."""
    config = {
        "project_settings": {
            "base_dir": base_dir,
            "project_name": project_name,
            "data_files": data_files,
            "project_dir": project_dir
        },
        "model_architecture": {
            "model_class": "StoryTellerTransformer",
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "dropout": dropout,
            "embed_dropout": embed_dropout,
            "dim_feedforward": dim_feedforward,
            "num_genres": num_genres,
            "d_c": d_c, 
            "d_prime_c": d_prime_c, 
            "d_R_h": d_R_h,
            "embedding_layer": f"nn.Embedding({vocab_size}, {d_model})",
            "output_layer": f"nn.Linear({d_model}, {vocab_size})",
            "decoder_layers": f"{num_layers} layers of DecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, d_c={d_c}, d_prime_c={d_prime_c}, d_R_h={d_R_h})",
            "normalization": "nn.LayerNorm(d_model)"
        },
        "chunking_tokenization": {
            "max_len": max_len,
            "padding_value": padding_value,
            "sliding_step": sliding_step,
            "max_chunks": max_chunks
        },
        "training_hyperparameters": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "start_epoch": start_epoch,
            "epochs": epochs,
            "max_grad_norm": max_grad_norm,
            "patience": patience,
            "use_optimizer": use_opt,
            "scheduler_type": scheduler_type,
            "warmup_epochs": warmup_epochs,
            "label_smoothing": label_smoothing,
            "grad_accum_steps": grad_accum_steps,
            "test_size": test_size,
            "rouge_sample_size": rouge_sample_size,
            "mixed_precision": mixed_precision,
            "plateau_factor": plateau_factor,
            "plateau_patience": plateau_patience,
            "use_optuna": use_optuna,
            "optuna_n_trials": optuna_n_trials
        },
        "generation_hyperparameters": {
            "split_ratio": split_ratio,
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "pad_idx": pad_idx,
            "eos_idx": eos_idx,
            "save_dis": save_dis
        },
        "device": str(device)
    }
    config_path = os.path.join(project_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Model configuration saved to {config_path}")
    logging.info(f"Model configuration saved to {config_path}")