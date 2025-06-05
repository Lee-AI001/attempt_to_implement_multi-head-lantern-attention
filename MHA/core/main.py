import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
from datetime import datetime
from tqdm import tqdm
from lion_pytorch import Lion
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer
import optuna
import datetime
from core.evaluate import evaluate
from dataloader.dataset import collate_fn, DataLoader
from core.model import StoryTellerTransformer

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, scheduler=None, warmup_scheduler=None, filename=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, filename or f"transformer_checkpoint_epoch_{epoch}_{timestamp}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if warmup_scheduler is not None:
        checkpoint['warmup_scheduler_state_dict'] = warmup_scheduler.state_dict()
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    print(f"✅ Model checkpoint saved: {checkpoint_path}")

def save_generated_story(story, prompt, index, output_file_path):
    try:
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(f"Prompt {index}:\n{prompt}\nStory:\n{story}\n\n")
        logging.info(f"Saved story {index}")
    except IOError as e:
        logging.error(f"Error saving story {index}: {e}")
        print(f"❌ Error saving story {index}: {e}")

def log_training_metrics(epoch, train_loss, dev_loss, perplexity, file_path, data_size=0, rouge_score=0.0, lr=0.0, epoch_time=0.0):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch:5d}   |   {train_loss:.4f}   |   {dev_loss:.4f}   |   {perplexity:.2f}   |   {data_size}   |   {rouge_score:.4f}   |   {lr:.6f}   |   {epoch_time:.2f}\n")
        logging.info(f"Metrics logged for epoch {epoch}: train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}, perplexity={perplexity:.2f}, data_size={data_size}, rouge_score={rouge_score:.4f}, lr={lr:.6f}, epoch_time={epoch_time:.2f}s")
        print(f"✅ Metrics logged for epoch {epoch}")
    except IOError as e:
        logging.error(f"Error logging metrics for epoch {epoch}: {e}")
        print(f"❌ Error logging metrics for epoch {epoch}: {e}")

def log_optuna_metrics(trial_number, val_loss, file_path, params, status="COMPLETED"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"--- Trial {trial_number} ---\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Status: {status}\n")
            f.write("Parameters:\n")
            for param_name, param_value in params.items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write("\n")
        logging.info(f"Optuna metrics logged for trial {trial_number}: val_loss={val_loss:.4f}, status={status}")
    except IOError as e:
        logging.error(f"Error logging Optuna metrics for trial {trial_number}: {e}")
        print(f"❌ Error logging Optuna metrics for trial {trial_number}: {e}")

def train_epoch(model, data_loader, criterion, optimizer, device, max_grad_norm, grad_accum_steps, mixed_precision, epoch):
    model.train()
    total_loss = 0
    total_batches = 0
    scaler = GradScaler() if mixed_precision != "fp32" else None
    optimizer.zero_grad()
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}")
    for i, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        
        with autocast(enabled=mixed_precision != "fp32", dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16):
            output = model(input_ids, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            loss = loss / grad_accum_steps
        
        if mixed_precision != "fp32":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % grad_accum_steps == 0:
            if mixed_precision != "fp32":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        total_batches += 1
        try:
            perplexity = math.exp(batch_loss)
        except OverflowError:
            perplexity = float('inf')
            logging.warning("Perplexity overflow. Setting to infinity.")
        progress_bar.set_postfix(loss=batch_loss, perplexity=perplexity)
    
    return total_loss / total_batches

def setup_optimizer_and_scheduler(model, use_opt, learning_rate, weight_decay, scheduler_type, epochs, warmup_epochs, plateau_factor, plateau_patience, sgdr_cycle_length=8, sgdr_min_lr=8e-6, sgdr_max_lr=2.5e-4):
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=weight_decay) if use_opt == "lion" else optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.1)
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    elif scheduler_type == "sgdr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=sgdr_cycle_length, eta_min=sgdr_min_lr, T_mult=1)
        warmup_scheduler = None
    elif scheduler_type == "sgd":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
        warmup_scheduler = None
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=plateau_factor, patience=plateau_patience, verbose=True)
        warmup_scheduler = None
    
    return optimizer, scheduler, warmup_scheduler

def objective(trial, train_dataset, dev_dataset, device, VOCAB_SIZE, PAD_IDX, num_genres, D_MODEL, NHEAD, DIM_FEEDFORWARD, NUM_LAYERS, OPTUNA_EPOCHS, USE_OPT, SCHEDULER_TYPE, PLATEAU_FACTOR, PLATEAU_PATIENCE, SGDR_CYCLE_LENGTH, SGDR_MIN_LR, SGDR_MAX_LR, MAX_GRAD_NORM, GRAD_ACCUM_STEPS, MIXED_PRECISION, optuna_metrics_file_path, model_class=StoryTellerTransformer, collate_fn=collate_fn):
    d_model = D_MODEL
    nhead = NHEAD
    dim_feedforward = DIM_FEEDFORWARD
    num_layers = NUM_LAYERS

    # Suggest training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 2e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.25)
    embed_dropout = trial.suggest_float('embed_dropout', 0.05, 0.1)
    warmup_epochs = trial.suggest_int('warmup_epochs', 0, 4)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX, nhead=nhead)
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=PAD_IDX, nhead=nhead)
    )

    # Model with MLA hyperparameters
    model = model_class(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        embed_dropout=embed_dropout,
        num_genres=num_genres
    ).to(device)

    # Optimizer and scheduler
    optimizer, scheduler, warmup_scheduler = setup_optimizer_and_scheduler(
        model, USE_OPT, learning_rate, weight_decay, SCHEDULER_TYPE, 5,
        warmup_epochs, PLATEAU_FACTOR, PLATEAU_PATIENCE,
        SGDR_CYCLE_LENGTH, SGDR_MIN_LR, SGDR_MAX_LR
    )

    # Loss function with tuned label_smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)

    # Training for OPTUNA_EPOCHS
    try:
        for epoch in range(OPTUNA_EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                MAX_GRAD_NORM, GRAD_ACCUM_STEPS, MIXED_PRECISION, epoch
            )
            val_metrics = evaluate(model, dev_loader, criterion, device, VOCAB_SIZE, PAD_IDX)
            val_loss = val_metrics['loss']
            val_perplexity = val_metrics['perplexity']

            # Check overfitting condition: train_loss should be greater than val_loss
            if train_loss <= val_loss:
                log_optuna_metrics(trial.number, float('inf'), optuna_metrics_file_path, trial.params, status="OVERFIT")
                return float('inf')  # Penalize trial if overfitting

            if SCHEDULER_TYPE == 'cosine':
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
            else:
                scheduler.step(val_loss)

            trial.report(val_perplexity, epoch)
            if trial.should_prune():
                log_optuna_metrics(trial.number, val_perplexity, optuna_metrics_file_path, trial.params, status="PRUNED")
                raise optuna.TrialPruned()

        # Log completed trial
        log_optuna_metrics(trial.number, val_perplexity, optuna_metrics_file_path, trial.params, status="COMPLETED")
        print(f"Trial {trial.number}: Val Perplexity = {val_perplexity:.4f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Params = {trial.params}")

        return val_perplexity

    except Exception as e:
        # Log failed trial
        log_optuna_metrics(trial.number, float('inf'), optuna_metrics_file_path, trial.params, status=f"FAILED: {str(e)}")
        raise

def plot_training_metrics(metrics_file_path, output_path):
    epochs, train_losses, dev_losses, perplexities, rouge_scores, lrs = [], [], [], [], [], []
    try:
        with open(metrics_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip header
                if line.strip() and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 6:
                        try:
                            epoch = int(parts[0].strip())
                            train_loss = float(parts[1].strip())
                            dev_loss = float(parts[2].strip())
                            perplexity = float(parts[3].strip())
                            rouge_score = float(parts[5].strip())
                            lr = float(parts[6].strip())
                            epochs.append(epoch)
                            train_losses.append(train_loss)
                            dev_losses.append(dev_loss)
                            perplexities.append(perplexity)
                            rouge_scores.append(rouge_score)
                            lrs.append(lr)
                        except ValueError:
                            continue
        if not epochs:
            logging.warning("No valid metrics data to plot")
            return

        plt.figure(figsize=(10, 12))
        
        # Plot Losses
        plt.subplot(4, 1, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, dev_losses, 'r-', label='Dev Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot Perplexity
        plt.subplot(4, 1, 2)
        plt.plot(epochs, perplexities, 'g-', label='Perplexity')
        plt.title('Validation Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.grid(True)
        plt.legend()

        # Plot ROUGE Scores
        plt.subplot(4, 1, 3)
        plt.plot(epochs, rouge_scores, 'm-', label='ROUGE Score')
        plt.title('ROUGE Score (Average of ROUGE-1 and ROUGE-L)')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE Score')
        plt.grid(True)
        plt.legend()

        # Plot Learning Rate
        plt.subplot(4, 1, 4)
        plt.plot(epochs, lrs, 'c-', label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Training metrics plot saved to {output_path}")
        print(f"✅ Training metrics plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to plot metrics: {e}")
        print(f"❌ Failed to plot metrics: {e}")