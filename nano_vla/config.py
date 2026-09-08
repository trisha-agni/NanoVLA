from pathlib import Path
import os

# --- FILE SYSTEM PATHS ---
ROOT = Path(__file__).resolve().parents[1]
MANUAL_DATA_DIR = ROOT / 'manual_dataset'
EXPERT_DATA_DIR = ROOT / 'expert_dataset'
ACTIVE_DATA_DIR = MANUAL_DATA_DIR
IMAGES_DIR = ACTIVE_DATA_DIR / 'images'
MANIFEST_FILE = 'manifest.json'
MANIFEST_PATH = os.path.join(ACTIVE_DATA_DIR, MANIFEST_FILE)

# --- ENVIRONMENT PARAMETERS ---
GRID_SZ = 10
WINDOW_SZ = 400

# --- ML BACKBONE SETTINGS ---
DEFAULT_MODEL_ID = 'gpt2'
DEFAULT_VISION_ENCODER = 'google/vit-base-patch16-224'
MAX_TEXT_LENGTH = 32
