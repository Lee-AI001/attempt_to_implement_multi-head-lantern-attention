
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
from dataloader.data_utils import clean_genre_label

class MoviePlotDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_tokens=512, step=256, max_chunks=24):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_tokens = max_tokens
        self.step = step
        self.max_chunks = max_chunks
        self.samples = self._tokenize_all()

    def _tokenize_all(self):
        all_chunks = []
        for story in self.data:
            tokens = self.tokenizer.encode(story["text"], out_type=int)
            primary_label = None
            for label in story.get("labels", []):
                cleaned_label = clean_genre_label(label)
                if cleaned_label in self.label2id:
                    primary_label = cleaned_label
                    break
            primary_label = primary_label or "Unknown"
            label_id = self.label2id.get(primary_label, self.tokenizer.piece_to_id("<Unknown>"))

            chunks = []
            for i in range(0, len(tokens), self.step):
                chunk = tokens[i:i + self.max_tokens]
                if len(chunk) < 2:
                    continue
                chunk = [label_id] + chunk
                chunks.append({"input_ids": chunk})
                if len(chunks) >= self.max_chunks:
                    break
            all_chunks.extend(chunks)
        return all_chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids[:-1],
            "target_ids": input_ids[1:]
        }

def pad_sequences(sequences, padding_value=0):
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

def create_mask(input_ids, nhead):
    seq_length = input_ids.shape[1]
    mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
    batch_size = input_ids.shape[0]
    return mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nhead, seq_length, seq_length).reshape(batch_size * nhead, seq_length, seq_length)

def create_padding_mask(input_ids, pad_idx=0):
    return input_ids.eq(pad_idx)

def collate_fn(batch, pad_idx=0, nhead=8):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    input_ids = pad_sequences(input_ids, padding_value=pad_idx)
    target_ids = pad_sequences(target_ids, padding_value=pad_idx)
    padding_mask = create_padding_mask(input_ids, pad_idx)
    attention_mask = create_mask(input_ids, nhead=nhead)
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "padding_mask": padding_mask
    }

def print_training_data_example(data_loader, tokenizer, pad_idx=0):
    batch = next(iter(data_loader))
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    attention_mask = batch["attention_mask"]
    padding_mask = batch["padding_mask"]
    print("=" * 60)
    print("📌 Input IDs (Padded Tokens)")
    print(input_ids)
    print(f"Shape: {input_ids.shape}")
    print("\n📌 Target IDs")
    print(target_ids)
    print(f"Shape: {target_ids.shape}")
    print("\n📌 Causal Attention Mask")
    print(attention_mask[0])
    print(f"Shape: {attention_mask.shape}")
    print("\n📌 Padding Mask")
    print(padding_mask)
    print(f"Shape: {padding_mask.shape}")
    sample_idx = 0
    sample_input = input_ids[sample_idx].tolist()
    decoded_text = tokenizer.decode(sample_input)
    print("\n📌 Decoded Sample Input")
    print(decoded_text)
    print("✅ Data loader sample validated!")
    logging.info("Data loader sample validated")