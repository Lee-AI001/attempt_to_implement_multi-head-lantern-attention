import os
import torch
import json
import time 
import torch.nn as nn
import sentencepiece as spm
import logging
import optuna
from torch.utils.data import DataLoader
import random
import datetime

from config import (
    PROJECT_NAME, BASE_DIR_OUT, DATA_FILES, VOCAB_SIZE, D_MODEL, NUM_LAYERS, NHEAD,
    DIM_FEEDFORWARD, EMBED_DROPOUT, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    WARMUP_EPOCHS, MAX_GRAD_NORM, LABEL_SMOOTHING, EPOCHS, START_EPOCH, SAVE_DIS,
    TEST_SIZE, BATCH_SIZE, GRAD_ACCUM_STEPS, SPLIT_RATIO, MAX_LEN, SLIDING_STEP,
    MAX_CHUNKS, PADDING_VALUE, TEMPERATURE, TOP_K, TOP_P, PAD_IDX, EOS_IDX,
    MAX_GEN_LEN, MIXED_PRECISION, USE_OPT, SCHEDULER_TYPE, PLATEAU_FACTOR,
    PLATEAU_PATIENCE, QUERIES, USE_OPTUNA, OPTUNA_N_TRIALS, STORAGE_DIR, CHECKPOINT_DIR,
    LOG_PATH, TRAIN_TXT_PATH, TOKENIZER_PREFIX, TOKENIZER_PATH, GENERATED_PATH,
    MODEL_ARCH_PATH, LABEL2ID_PATH, SGDR_CYCLE_LENGTH, SGDR_MIN_LR, SGDR_MAX_LR,
    ROUGE_SAMPLE_SIZE, OPTUNA_EPOCHS
)
from dataloader.data_utils import (
    load_and_clean_data, split_data, write_train_texts, extract_genre_labels,
    save_model_config
)
from dataloader.dataset import MoviePlotDataset, collate_fn, print_training_data_example
from core.model import StoryTellerTransformer
from core.main import (
    train_epoch, setup_optimizer_and_scheduler, save_checkpoint,
    log_training_metrics, save_generated_story, log_optuna_metrics,
    objective, plot_training_metrics
)
from core.evaluate import evaluate, compute_rouge_with_subset
from core.generate import generate_multiple_stories

# Debug: Torch + CUDA
print("✅ Torch:", torch.__version__)
print("✅ CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths and Logging
project_dir = STORAGE_DIR
checkpoint_dir = CHECKPOINT_DIR
os.makedirs(project_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Training started.")

train_txt_path = TRAIN_TXT_PATH
tokenizer_prefix = TOKENIZER_PREFIX
tokenizer_model_path = TOKENIZER_PATH
output_file_path = GENERATED_PATH
metrics_file_path = os.path.join(project_dir, "training_metrics.txt")
optuna_metrics_file_path = os.path.join(project_dir, "optuna_metrics.txt")
model_config_path = MODEL_ARCH_PATH

print(f"✅ Project Directory: {project_dir}")
print(f"✅ Checkpoints: {checkpoint_dir}")
print(f"✅ Tokenizer Path: {tokenizer_model_path}")

# Load and Split Data
cleaned_data = load_and_clean_data(DATA_FILES, max_size_mb=TEST_SIZE)
train_data, dev_data = split_data(cleaned_data, SPLIT_RATIO)
write_train_texts(train_data, train_txt_path)
print(f"✅ Data Loaded: {TEST_SIZE} MB")

# Extract genre labels and compute num_genres
genre_labels = extract_genre_labels(cleaned_data)
num_genres = len(genre_labels)
print(f"✅ Number of genres: {num_genres}")

# Save model configuration
save_model_config(
    project_dir, BASE_DIR_OUT, PROJECT_NAME, DATA_FILES, VOCAB_SIZE, NUM_LAYERS,
    D_MODEL, NHEAD, DROPOUT, EMBED_DROPOUT, DIM_FEEDFORWARD, MAX_LEN, PADDING_VALUE,
    SLIDING_STEP, MAX_CHUNKS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, START_EPOCH,
    EPOCHS, MAX_GRAD_NORM, PATIENCE, USE_OPT, SCHEDULER_TYPE, WARMUP_EPOCHS,
    SPLIT_RATIO, MAX_GEN_LEN, TEMPERATURE, PAD_IDX, EOS_IDX, SAVE_DIS, device,
    LABEL_SMOOTHING, GRAD_ACCUM_STEPS, TEST_SIZE, ROUGE_SAMPLE_SIZE, MIXED_PRECISION,
    PLATEAU_FACTOR, PLATEAU_PATIENCE, USE_OPTUNA, OPTUNA_N_TRIALS,
    num_genres
)

# Tokenizer Training
if not os.path.exists(tokenizer_model_path):
    print("🚀 Training SentencePiece tokenizer...")
    user_defined_symbols = ",".join([f"<{genre}>" for genre in genre_labels])
    spm.SentencePieceTrainer.Train(
        input=train_txt_path,
        model_prefix=tokenizer_prefix,
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="bpe",
        user_defined_symbols=user_defined_symbols,
        pad_id=PAD_IDX,
        unk_id=1,
        bos_id=2,
        eos_id=EOS_IDX
    )
    print("✅ Tokenizer trained successfully.")
    logging.info("Tokenizer trained successfully")
else:
    print("🔁 Tokenizer already exists, skipping training.")
    logging.info("Tokenizer already exists, skipping training")

sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)
print("✅ Tokenizer loaded with vocab size:", sp.get_piece_size())
logging.info(f"Tokenizer loaded with vocab size: {sp.get_piece_size()}")

label2id = {genre: sp.piece_to_id(f"<{genre}>") for genre in genre_labels}
with open(LABEL2ID_PATH, "w", encoding="utf-8") as f:
    json.dump(label2id, f, indent=4)
print("✅ Label2ID mapping:", label2id)
logging.info(f"Label2ID mapping: {label2id}")

for genre in genre_labels:
    token = f"<{genre}>"
    if sp.piece_to_id(token) < 0:
        raise ValueError(f"Genre token {token} not found in tokenizer vocabulary")
print("✅ All genre tokens verified.")
logging.info("All genre tokens verified")

train_dataset = MoviePlotDataset(
    data=train_data, tokenizer=sp, label2id=label2id, max_tokens=MAX_LEN,
    step=SLIDING_STEP, max_chunks=MAX_CHUNKS
)
dev_dataset = MoviePlotDataset(
    data=dev_data, tokenizer=sp, label2id=label2id, max_tokens=MAX_LEN,
    step=SLIDING_STEP, max_chunks=MAX_CHUNKS
)
data_size = len(train_dataset.samples)
print(f"✅ Chunked Data Size: {data_size} samples (Data Limit: {TEST_SIZE} MB)")

# Debug DataLoader
test_loader = DataLoader(
    MoviePlotDataset(
        train_data[:10], sp, label2id, max_tokens=MAX_LEN, step=SLIDING_STEP,
        max_chunks=MAX_CHUNKS
    ),
    batch_size=2, shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX, nhead=NHEAD)
)
print_training_data_example(test_loader, sp, PAD_IDX)

# Run Optuna optimization if enabled
best_params = {}
if USE_OPTUNA:
    print("🚀 Running Optuna hyperparameter optimization...")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(
        lambda trial: objective(
            trial, train_dataset, dev_dataset, device, VOCAB_SIZE, PAD_IDX, num_genres,
            D_MODEL, NHEAD, DIM_FEEDFORWARD, NUM_LAYERS, OPTUNA_EPOCHS, USE_OPT,
            SCHEDULER_TYPE, PLATEAU_FACTOR, PLATEAU_PATIENCE, SGDR_CYCLE_LENGTH,
            SGDR_MIN_LR, SGDR_MAX_LR, MAX_GRAD_NORM, GRAD_ACCUM_STEPS, MIXED_PRECISION
        ),
        n_trials=OPTUNA_N_TRIALS
    )
    best_params = study.best_params
    print(f"✅ Best hyperparameters: {best_params}")
    print(f"✅ Best validation loss: {study.best_value}")
    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Best validation loss: {study.best_value}")

# Update config with best hyperparameters or use defaults
d_model = best_params.get('d_model', D_MODEL)
nhead = best_params.get('nhead', NHEAD)
dim_feedforward = 4 * d_model  # Fixed multiplier
learning_rate = best_params.get('learning_rate', LEARNING_RATE)
weight_decay = best_params.get('weight_decay', WEIGHT_DECAY)
batch_size = best_params.get('batch_size', BATCH_SIZE)
num_layers = best_params.get('num_layers', NUM_LAYERS)
dropout = best_params.get('dropout', DROPOUT)
embed_dropout = best_params.get('embed_dropout', EMBED_DROPOUT)
warmup_epochs = best_params.get('warmup_epochs', WARMUP_EPOCHS)
label_smoothing = best_params.get('label_smoothing', LABEL_SMOOTHING)

# Model Setup with final parameters
model = StoryTellerTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    embed_dropout=embed_dropout,
    num_genres=num_genres,
).to(device)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX, nhead=model.nhead)
)
dev_loader = DataLoader(
    dev_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX, nhead=model.nhead)
)

# Optimizer and Scheduler
optimizer, scheduler, warmup_scheduler = setup_optimizer_and_scheduler(
    model, USE_OPT, learning_rate, weight_decay, SCHEDULER_TYPE, EPOCHS,
    warmup_epochs, PLATEAU_FACTOR, PLATEAU_PATIENCE,
    SGDR_CYCLE_LENGTH, SGDR_MIN_LR, SGDR_MAX_LR
)

# Loss Function
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)

# Initialize Metrics File
if not os.path.exists(metrics_file_path):
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write("Training Metrics\nEpoch  |  Train Loss  |  Dev Loss  |  Perplexity  |  Data Size  |  ROUGE Score  |  LR  |  Time (s)  |  \ovo/\n" + "-" * 90 + "\n")

# Training Loop
start_epoch = START_EPOCH
epochs_no_improve = 0
best_val_loss = float('inf')
best_model_path = os.path.join(checkpoint_dir, "transformer_best_model.pth")

checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("transformer_checkpoint") and f.endswith(".pth")]
if checkpoint_files:
    try:
        # Extract epoch numbers from filenames
        epoch_numbers = []
        for f in checkpoint_files:
            try:
                epoch_num = int(f.split('_')[-1].split('.')[0])
                epoch_numbers.append((epoch_num, f))
            except ValueError:
                continue
        if epoch_numbers:
            latest_epoch, latest_checkpoint = max(epoch_numbers)
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if SCHEDULER_TYPE == "cosine":
                scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
                warmup_scheduler.load_state_dict(checkpoint.get('warmup_scheduler_state_dict', warmup_scheduler.state_dict()))
            elif SCHEDULER_TYPE in ["sgdr", "sgd"]:
                scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
            else:
                scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            logging.info(f"Resumed training from {checkpoint_path} at epoch {checkpoint['epoch']} to start at epoch {start_epoch}")
            print(f"Resumed training from epoch {checkpoint['epoch']} with best validation loss {best_val_loss:.4f}, starting at epoch {start_epoch}")
        else:
            logging.warning("No valid checkpoint files found. Starting from scratch.")
            start_epoch = START_EPOCH
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
        print(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = START_EPOCH
else:
    start_epoch = START_EPOCH

for epoch in range(start_epoch, EPOCHS):
    epoch_start_time = time.time()
    
    avg_train_loss = train_epoch(
        model, train_loader, criterion, optimizer, device, MAX_GRAD_NORM,
        GRAD_ACCUM_STEPS, MIXED_PRECISION, epoch
    )
    
    val_metrics = evaluate(model, dev_loader, criterion, device, VOCAB_SIZE, PAD_IDX)
    val_loss = val_metrics["loss"]
    val_perplexity = val_metrics["perplexity"]
    
    current_lr = optimizer.param_groups[0]['lr']
    
    if SCHEDULER_TYPE == "cosine":
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
    elif SCHEDULER_TYPE == "sgdr":
        scheduler.step()
    elif SCHEDULER_TYPE == "sgd":
        scheduler.step()
    else:
        scheduler.step(val_loss)
    
    print(f"Model nhead: {model.nhead}")
    print(f"Number of queries: {len(QUERIES)}")
    results = generate_multiple_stories(
        model, sp, QUERIES, label2id=label2id, genres=None, max_length=MAX_GEN_LEN,
        temperature=TEMPERATURE, pad_idx=PAD_IDX, eos_idx=EOS_IDX, top_k=TOP_K, top_p=TOP_P
    )
    generated_stories = [story for _, _, story in results]
    reference_texts = [sample["text"] for sample in random.sample(dev_data, min(ROUGE_SAMPLE_SIZE, len(dev_data)))]
    rouge_score = compute_rouge_with_subset(
        model, sp, dev_data, label2id, num_samples=ROUGE_SAMPLE_SIZE,
        max_length=MAX_GEN_LEN, temperature=TEMPERATURE, pad_idx=PAD_IDX,
        eos_idx=EOS_IDX, top_k=TOP_K, top_p=TOP_P
    )
    epoch_time = time.time() - epoch_start_time
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    print(f"  Val Perplexity: {val_perplexity:.2f}")
    print(f"  Chunked Data Size: {data_size}")
    print(f"  ROUGE Score: {rouge_score:.4f}")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Epoch Time: {epoch_time:.2f}s")
    logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.2f}, Data Size = {data_size}, ROUGE Score = {rouge_score:.4f}, LR = {current_lr:.6f}, Epoch Time = {epoch_time:.2f}s")
    log_training_metrics(epoch + 1, avg_train_loss, val_loss, val_perplexity, metrics_file_path, data_size, rouge_score, current_lr, epoch_time)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_path = os.path.join(checkpoint_dir, "transformer_best_model.pth")
        save_checkpoint(
            model, optimizer, epoch + 1, avg_train_loss, val_loss, checkpoint_dir,
            scheduler=scheduler, warmup_scheduler=warmup_scheduler if SCHEDULER_TYPE == "cosine" else None,
            filename="transformer_best_model.pth"
        )
        logging.info(f"Saved best model with val loss {best_val_loss:.4f}")
        print(f"✅ Saved best model: {best_model_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            print("Early stopping triggered!")
            break
    
    if (epoch + 1) % SAVE_DIS == 0:
        save_checkpoint(
            model, optimizer, epoch + 1, avg_train_loss, val_loss, checkpoint_dir,
            scheduler=scheduler, warmup_scheduler=warmup_scheduler if SCHEDULER_TYPE == "cosine" else None
        )
    
    print(f"\n===== Generated Stories for Epoch {epoch + 1} =====")
    with open(output_file_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== Generated Stories for Epoch {epoch + 1} =====\n")
    for i, (prompt, genre, story) in enumerate(results, 1):
        print(f"\nPrompt {i}: {prompt[:100]}...")
        print(f"Story {i}: {story[:250]}...")
        save_generated_story(story, prompt, i, output_file_path)
        logging.info(f"Generated story {i} for epoch {epoch+1}: {story[:100]}...")

# Save final model
final_model_path = os.path.join(checkpoint_dir, "transformer_final_model.pth")
save_checkpoint(
    model, optimizer, EPOCHS, avg_train_loss, val_loss, checkpoint_dir,
    scheduler=scheduler, warmup_scheduler=warmup_scheduler if SCHEDULER_TYPE == "cosine" else None,
    filename="transformer_final_model.pth"
)
logging.info(f"Saved final model: {final_model_path}")
print(f"✅ Final model saved: {final_model_path}")

print("\n===== Final Generated Stories =====")
final_results = generate_multiple_stories(
    model, sp, QUERIES, label2id=label2id, genres=None, max_length=MAX_GEN_LEN,
    temperature=TEMPERATURE, pad_idx=PAD_IDX, eos_idx=EOS_IDX, top_k=TOP_K, top_p=TOP_P
)
with open(output_file_path, "a", encoding="utf-8") as f:
    f.write("\n===== Final Generated Stories =====\n")
for i, (prompt, genre, story) in enumerate(final_results, 1):
    print(f"\nPrompt {i}: {prompt}")
    print(f"Story {i}:\n{story}\n")
    save_generated_story(story, prompt, i, output_file_path)
    logging.info(f"Final story {i}: {story[:100]}...")

plot_training_metrics(metrics_file_path, os.path.join(project_dir, "training_metrics_plot.png"))