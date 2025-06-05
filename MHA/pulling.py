import torch
import os
import argparse
import logging
import json
from datetime import datetime
from core.model import StoryTellerTransformer
from core.generate import generate_story
from dataloader.tokenizer_utils import load_tokenizer

# Configure logging to output only to console
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
    logging.StreamHandler()
])

# Define constants from model_config.json -> generation_hyperparameters
PAD_IDX = 0
EOS_IDX = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_tokenizer_type(folder_path):
    """Detect tokenizer type based on files in the folder."""
    logging.debug(f"Checking tokenizer type in {folder_path}")
    model_path = os.path.join(folder_path, "story_tokenizer.model")
    if os.path.exists(model_path):
        logging.info(f"Found SentencePiece tokenizer at {model_path}")
        return "spm"
    logging.error(f"No tokenizer found at {model_path}")
    raise ValueError("Unable to detect tokenizer type: missing tokenizer files")

def load_genre_labels(label2id_path):
    """Load genre labels and LABEL2ID from a JSON file."""
    logging.debug(f"Loading genre labels from {label2id_path}")
    try:
        with open(label2id_path, 'r', encoding='utf-8') as f:
            label2id = json.load(f)
        genres = sorted(label2id.keys())
        num_genres = len(genres)
        logging.info(f"Loaded {num_genres} genres: {genres}")
        return genres, label2id, num_genres
    except Exception as e:
        logging.error(f"Failed to load genre labels: {e}")
        raise RuntimeError(f"Failed to load genre labels: {e}")

def load_model_config(config_path):
    """Load model configuration from model_config.json."""
    logging.debug(f"Loading model configuration from {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Loaded model configuration content: {json.dumps(config, indent=2)}")
        
        # Extract model architecture and project settings
        if "model_architecture" not in config:
            raise ValueError("Missing 'model_architecture' section in model_config.json")
        if "project_settings" not in config:
            raise ValueError("Missing 'project_settings' section in model_config.json")
        
        model_arch = config["model_architecture"]
        project_settings = config["project_settings"]
        
        # Check required keys in model_architecture
        required_keys = ["vocab_size", "num_layers", "d_model", "nhead", "dim_feedforward", "num_genres"]
        missing_keys = [key for key in required_keys if key not in model_arch]
        if missing_keys:
            logging.error(f"Missing required keys in model_architecture: {missing_keys}")
            raise ValueError(f"Missing key(s) in model_architecture: {missing_keys}")
        
        # Check project_name in project_settings
        if "project_name" not in project_settings:
            logging.error("Missing 'project_name' in project_settings")
            raise ValueError("Missing 'project_name' in project_settings")
        
        # Create config dictionary for model initialization
        model_config = {
            "vocab_size": model_arch["vocab_size"],
            "num_layers": model_arch["num_layers"],
            "d_model": model_arch["d_model"],
            "nhead": model_arch["nhead"],
            "dim_feedforward": model_arch["dim_feedforward"],
            "num_genres": model_arch["num_genres"],
            "project_name": project_settings["project_name"]
        }
        return model_config
    except Exception as e:
        logging.error(f"Failed to load model configuration: {e}")
        raise RuntimeError(f"Failed to load model configuration: {e}")

def load_model(checkpoint_dir, model_config):
    """Load a model checkpoint based on user-selected file."""
    logging.debug(f"Loading model with configuration: {model_config}")
    try:
        model = StoryTellerTransformer(
            vocab_size=model_config["vocab_size"],
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=0,  # Override to 0 as specified
            embed_dropout=0,  # Override to 0 as specified
            num_genres=model_config["num_genres"]
        )
        logging.info(f"Initialized model with {model_config['num_genres']} genres")
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Failed to initialize model: {e}")

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        logging.error(f"No checkpoints found in {checkpoint_dir}")
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    priority_files = ["transformer_best_model.pth", "transformer_final_model.pth"]
    available_priority = [f for f in priority_files if f in checkpoint_files]
    other_files = [f for f in checkpoint_files if f not in available_priority]
    sorted_files = available_priority + sorted(other_files)

    print("\n📂 Available checkpoints:")
    for i, file in enumerate(sorted_files, 1):
        checkpoint_path = os.path.join(checkpoint_dir, file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            epoch = checkpoint.get("epoch", "N/A")
            val_loss = checkpoint.get("val_loss", "N/A")
            print(f"{i}: {file} (Epoch: {epoch}, Val Loss: {val_loss:.4f})")
        except Exception as e:
            print(f"{i}: {file} (Metadata unavailable: {e})")
    
    while True:
        try:
            choice = input("\n🔍 Select checkpoint by number (1-{}): ".format(len(sorted_files))).strip()
            choice = int(choice)
            if choice < 1 or choice > len(sorted_files):
                print(f"⚠️ Invalid choice. Please enter a number between 1 and {len(sorted_files)}. 😊")
                continue
            
            checkpoint_path = os.path.join(checkpoint_dir, sorted_files[choice - 1])
            logging.debug(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
            checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
            model_state_dict.update(checkpoint_state_dict)
            model.load_state_dict(model_state_dict, strict=False)  # Allow partial loading
            
            genre_frozen = checkpoint.get("genre_frozen", False)
            if genre_frozen:
                for param in model.genre_head.parameters():
                    param.requires_grad = False
                logging.info("Restored genre_frozen status: True")
            
            model.eval()
            model = model.to(DEVICE)
            logging.info(f"Successfully loaded checkpoint: {checkpoint_path}")
            return model
        except ValueError:
            print("⚠️ Invalid input. Please enter a number. 😊")
        except Exception as e:
            logging.error(f"Checkpoint load error: {e}")
            print(f"❌ Checkpoint load error: {e}. Please try another checkpoint.")
            continue

def story_to_markdown(query, body, top_genre=None, project_name=""):
    """Convert story to Markdown format with user query and top predicted genre."""
    logging.debug(f"Formatting story to Markdown, query: {query[:50]}..., genre: {top_genre}")
    body = body.replace('\n', '\n\n')
    body = body.replace('--', '—')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_template = """## Story (Generated on {timestamp})
**User Prompt:**  
{query}

**{project_name} (Genre: {genre}):**  
{body}

---
"""
    return markdown_template.format(
        timestamp=timestamp,
        query=query,
        project_name=project_name,
        genre=top_genre if top_genre else "None",
        body=body
    )

def interactive_loop(model, tokenizer, label2id, output_file, min_len=100, max_length=250, project_name=""):
    """Interactive loop for generating stories with user-selected genres."""
    print("🧠 Welcome to the Interactive Story Generator! 😊")
    print("Type your story prompt, 'exit' to quit, or 'help' for commands.")
    prompt_history = []
    genres = list(label2id.keys())  # Available genres from label2id

    # Validate tokenizer and model vocab size
    tokenizer_vocab_size = tokenizer.get_piece_size()
    model_vocab_size = model.lm_head.out_features
    if tokenizer_vocab_size != model_vocab_size:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) differs from model vocab size ({model_vocab_size})")
        print(f"⚠️ Warning: Tokenizer vocab size ({tokenizer_vocab_size}) differs from model vocab size ({model_vocab_size})")

    while True:
        query = input("\n📝 Your story prompt: ").strip()
        if query.lower() == "exit":
            print("👋 Thanks for creating stories! Goodbye! 😊")
            break
        elif query.lower() == "help":
            print("\n📚 Commands:")
            print("- Enter a prompt to generate a story.")
            print("- 'exit': Quit the generator.")
            print("- 'help': Show this help message.")
            print("- 'history': View previous prompts in this session.")
            print("- 'tune': Adjust generation settings (e.g., temperature).")
            continue
        elif query.lower() == "history":
            if not prompt_history:
                print("📜 No prompts yet in this session.")
            else:
                print("\n📜 Prompt History:")
                for i, prompt in enumerate(prompt_history, 1):
                    print(f"{i}: {prompt[:50]}...")
            continue
        elif query.lower() == "tune":
            use_custom = "yes"
        else:
            prompt_history.append(query)
            # Genre selection with numbers
            print("\n🎭 Available genres:")
            for i, genre in enumerate(genres, 1):
                print(f"{i}: {genre}")
            genre_choice = input("\n🔍 Select genre by number (1-{}, or press Enter to skip): ".format(len(genres))).strip()
            selected_genre = None
            if genre_choice:
                try:
                    genre_idx = int(genre_choice) - 1
                    if 0 <= genre_idx < len(genres):
                        selected_genre = genres[genre_idx]
                        print(f"✅ Selected genre: {selected_genre}")
                    else:
                        print("⚠️ Invalid genre number. Proceeding without genre.")
                except ValueError:
                    print("⚠️ Invalid input. Proceeding without genre.")
            else:
                print("✅ Proceeding without genre.")
            use_custom = input("\n🔧 Tune settings? (yes/no, default=no): ").lower()

        if use_custom in ["yes", "y"]:
            try:
                temp = float(input("🌡️ Temperature (default=1.0): ") or 1.0)
                top_k = int(input("🎯 Top-k (default=50, 0 to disable): ") or 50)
                top_p = float(input("🎲 Top-p (default=0.9): ") or 0.9)
                max_len = int(input("📏 Max length (default=512): ") or 512)
                min_len = int(input("📐 Min length (default=100): ") or 100)
                if min_len >= max_len:
                    print("⚠️ Min length must be less than max length. Using defaults. 😊")
                    max_len = 250
                    min_len = 100
            except ValueError:
                print("⚠️ Invalid input. Using default settings. 😊")
                temp = 1.0
                top_k = 50
                top_p = 0.9
                max_len = 512
                min_len = 100
        else:
            temp = 1.0
            top_k = 50
            top_p = 0.9
            max_len = 512
            min_len = 100
            print("✅ Using default settings: temperature=1.0, top_k=50, top_p=0.9, max_length=512, min_length=100")

        logging.debug(f"Appending story to {output_file}")
        try:
            # Generate story with selected genre
            logging.debug(f"Generating story with max_length={max_len}, min_length={min_len}, genre={selected_genre}")
            with torch.amp.autocast('cuda', enabled=DEVICE.type == "cuda"):
                story = generate_story(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    label2id=label2id,
                    genre=selected_genre,
                    max_length=max_len,
                    temperature=temp,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    pad_idx=PAD_IDX,
                    eos_idx=EOS_IDX
                )

            torch.cuda.empty_cache()
            logging.debug(f"Generated story: {story[:100]}...")

            # Validate story content
            if not story or story.strip() == "":
                logging.error("Generated story is empty or invalid")
                print("❌ Error: Generated story is empty. Try another prompt. 😊")
                continue

            # Remove query from story if present
            query_lower = query.lower().strip()
            story_lower = story.lower()
            if story_lower.startswith(query_lower):
                story = story[len(query):].lstrip()
                logging.debug(f"Removed query prefix from story: {story[:100]}...")

            # Generate and append Markdown
            markdown_output = story_to_markdown(query, story, selected_genre, project_name)
            logging.debug(f"Markdown output: {markdown_output[:100]}...")

            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(markdown_output)
                logging.info(f"Successfully appended story to {output_file} (Size: {os.path.getsize(output_file)/1024:.2f} KB)")
                print(f"✅ Story saved to {output_file} 🎉")
            except IOError as e:
                logging.error(f"File append error for {output_file}: {e}")
                print(f"❌ Error saving to {output_file}: {e}. Try again. 😊")
                continue

            # Print story to console
            print("\n📜 Your Story:")
            print("=" * 60)
            print(markdown_output)
            print("=" * 60)
        except Exception as e:
            logging.error(f"Error during story generation: {e}")
            print(f"❌ Error generating story: {e}. Please try again. 😊")
            continue

def main():
    parser = argparse.ArgumentParser(description="Interactive Story Generator")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum story length in tokens")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum story length in tokens")
    args = parser.parse_args()

    # Define storage directory
    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    if not os.path.exists(storage_dir):
        logging.error(f"Storage directory not found: {storage_dir}")
        raise FileNotFoundError(f"Storage directory not found: {storage_dir}")

    # List project folders in storage
    project_folders = [f for f in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, f))]
    if not project_folders:
        logging.error("No project folders found in storage directory")
        raise FileNotFoundError("No project folders found in storage directory")

    # Prompt user to select a project folder
    print("\n📂 Available project folders:")
    for i, folder in enumerate(sorted(project_folders), 1):
        print(f"{i}: {folder}")
    
    while True:
        try:
            folder_choice = input("\n🔍 Select project folder by number (1-{}): ".format(len(project_folders))).strip()
            folder_choice = int(folder_choice)
            if folder_choice < 1 or folder_choice > len(project_folders):
                print(f"⚠️ Invalid choice. Please enter a number between 1 and {len(project_folders)}. 😊")
                continue
            selected_folder = project_folders[folder_choice - 1]
            selected_folder_path = os.path.join(storage_dir, selected_folder)
            break
        except ValueError:
            print("⚠️ Invalid input. Please enter a number. 😊")
            continue

    # Check for required files in the project folder
    config_path = os.path.join(selected_folder_path, "model_config.json")
    if not os.path.exists(config_path):
        logging.error(f"Model config file not found: {config_path}")
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    # Load model configuration
    model_config = load_model_config(config_path)
    project_name = model_config["project_name"]

    # Load genre labels
    label2id_path = os.path.join(selected_folder_path, "label2id.json")
    if not os.path.exists(label2id_path):
        logging.error(f"Genre labels file not found: {label2id_path}")
        raise FileNotFoundError(f"Genre labels file not found: {label2id_path}")
    genres, label2id, num_genres = load_genre_labels(label2id_path)

    # Load tokenizer
    tokenizer_path = os.path.join(selected_folder_path, "story_tokenizer.model")
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer file not found: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    # Load model from checkpoints
    checkpoint_dir = os.path.join(selected_folder_path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        logging.error(f"No 'checkpoints' subfolder found in {selected_folder}")
        raise FileNotFoundError(f"No 'checkpoints' subfolder found in {selected_folder}")
    model = load_model(checkpoint_dir, model_config)

    # Create output file in the project folder
    output_file = os.path.join(selected_folder_path, "output.md")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Story Generator Chat Log\n\n")
            f.write("Welcome to the interactive story generator! 🎉\n")
            f.write("Generated stories are saved here.\n\n")
        logging.info(f"Initialized {output_file}")
        print(f"✅ Created/Reset {output_file} 📝")
    except Exception as e:
        logging.error(f"Failed to create {output_file}: {e}")
        raise RuntimeError(f"Failed to create {output_file}: {e}")

    # Start interactive loop
    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        output_file=output_file,
        min_len=args.min_len,
        max_length=args.max_length,
        project_name=project_name
    )

if __name__ == "__main__":
    main()