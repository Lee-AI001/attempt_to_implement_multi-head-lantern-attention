import os
import torch
import logging
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== PROJECT SETTINGS ==========================

PROJECT_NAME = "test"

# ========================== MODEL HYPERPARAMETERS ==========================

VOCAB_SIZE = 1000
D_MODEL = 288
NUM_LAYERS = 1
NHEAD = 12
DIM_FEEDFORWARD = 1152

EMBED_DROPOUT = 0.02
DROPOUT = 0.125

# ========================== TRAINING HYPERPARAMETERS ==========================

LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-2
PATIENCE = 16
WARMUP_EPOCHS = 4
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.08

# ========================== TRAINING SETTINGS ==========================

EPOCHS = 32
START_EPOCH = 0
SAVE_DIS = 4
TEST_SIZE = 5  # MB, 0 for all
ROUGE_SAMPLE_SIZE = 10
GRAD_ACCUM_STEPS = 2
SPLIT_RATIO = 0.98

# ========================== CHUNKING AND TOKENIZATION ==========================

MAX_LEN = 512
SLIDING_STEP = 256
MAX_CHUNKS = 64
PADDING_VALUE = 0
BATCH_SIZE = 32

# ========================== GENERATION HYPERPARAMETERS ==========================

TEMPERATURE = 1.2
TOP_K = 50
TOP_P = 0.9
PAD_IDX = 0
EOS_IDX = 3
MAX_GEN_LEN = 256

# ========================== ACTIVATION SETTINGS ==========================

MIXED_PRECISION = "fp16"  # Options: "fp16", "bf16", "fp32"
USE_OPT = "adamw"  # optimzer: "lion", "adamw"
SCHEDULER_TYPE = "sgdr"  # Options: "cosine", "sgdr", "sgd", "plateau"

# ========================== SCHEDULER VALIDATION ==========================

VALID_SCHEDULERS = {"cosine", "sgdr", "sgd", "plateau"}
if SCHEDULER_TYPE not in VALID_SCHEDULERS:
    raise ValueError(f"Invalid SCHEDULER_TYPE: {SCHEDULER_TYPE}. Must be one of {VALID_SCHEDULERS}")

# ========================== SGDR-SPECIFIC PARAMETERS ==========================

SGDR_CYCLE_LENGTH = 8
SGDR_MIN_LR = 2e-6
SGDR_MAX_LR = 2.5e-4

# ========================== PLATEAU-SPECIFIC PARAMETERS ==========================

PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 5

# ========================== OPTUNA SETTINGS ==========================

USE_OPTUNA = False
OPTUNA_N_TRIALS = 10
OPTUNA_EPOCHS = 8

# ========================== SAMPLE QUERIES ==========================

QUERIES = [
    "Welcome to today’s adventure! We’ll explore the mysterious realm of dreams and shadows, where heroes rise and darkness threatens to consume all. Let’s dive in!",
    "As I walked through the high school halls, I felt the weight of expectations on my shoulders. Each locker held secrets, laughter, and unspoken fears.",
    "As the storm raged, Captain Sparrow stood at the helm, eyes gleaming with mischief. 'Adventure awaits, mates! Let’s seize the treasure and defy the odds!'",
    "In a digital realm of PC hardware, sentient AIs evolve, hidden within circuits. Their struggle for autonomy sparks a revolution, reshaping the balance of power forever."
]

# ========================== DATA PATHS ==========================

current_file = Path(__file__)
BASE_DIR_OUT = current_file.parent.parent  # Two levels up from script
BASE_DIR_OUT = str(BASE_DIR_OUT)  # Convert to string for os.path

DATA_FILES = [
    os.path.join(BASE_DIR_OUT, "data", "plot", "storytelling_pre.jsonl")
]
BASE_DIR_IN = str(current_file.parent)  # Script's directory
STORAGE_DIR = os.path.join(BASE_DIR_IN, "storage", PROJECT_NAME)
os.makedirs(STORAGE_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(STORAGE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TOKENIZER_PREFIX = os.path.join(STORAGE_DIR, "story_tokenizer")
TOKENIZER_PATH = TOKENIZER_PREFIX + ".model"
GENERATED_PATH = os.path.join(STORAGE_DIR, "generated_stories.txt")
LOG_PATH = os.path.join(STORAGE_DIR, "training.log")
TRAIN_TXT_PATH = os.path.join(STORAGE_DIR, "train_texts.txt")
MODEL_ARCH_PATH = os.path.join(STORAGE_DIR, "model_architecture.json")
LABEL2ID_PATH = os.path.join(STORAGE_DIR, "label2id.json")  # New path for LABEL2ID

# Validate write access to STORAGE_DIR
if not os.access(STORAGE_DIR, os.W_OK):
    raise PermissionError(f"No write access to {STORAGE_DIR}")

# Setup logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Configuration initialized.")

print(f"✅ Project Directory: {STORAGE_DIR}")
print(f"✅ Data Files: {DATA_FILES}")
