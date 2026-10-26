import os
from pathlib import Path

# ====================================================================
# ROOT DIRECTORY
# ====================================================================
BASE_DIR = Path(__file__).resolve().parent

# ====================================================================
# DATA & DICTIONARY PATHS (RELATIVE)
# ====================================================================
DATA_DIR = Path(os.getenv("VSL_DATA_DIR", BASE_DIR / "data"))

# Dictionary mapping: token (Vietnamese) -> video_id / label
MAPPING_FILE = Path(os.getenv("VSL_MAPPING_FILE", DATA_DIR / "dictionary" / "test.json"))

# Pose landmarks directory (.pose binary files)
POSE_DIR = Path(os.getenv("VSL_POSE_DIR", DATA_DIR / "poses"))
SAMPLE_POSE_DIR = DATA_DIR / "sample_poses"

# Fallback to sample_poses if poses directory is empty or missing .pose files
if not (POSE_DIR.exists() and any(POSE_DIR.glob("*.pose"))) and SAMPLE_POSE_DIR.exists():
    POSE_DIR = SAMPLE_POSE_DIR

# ====================================================================
# TEXT-TO-GLOSS (ViT5) MODEL PATH
# ====================================================================
MODEL_DIR = Path(os.getenv("VSL_MODEL_DIR", BASE_DIR / "models" / "vit5_sign_grammar"))

# ====================================================================
# HYBRID SEMANTIC SEARCH PATHS (PhoBERT + FAISS)
# ====================================================================
FAISS_INDEX_PATH = Path(os.getenv("VSL_FAISS_INDEX", DATA_DIR / "semantic" / "phrases.index"))
SENTENCES_TXT_PATH = Path(os.getenv("VSL_SENTENCES_TXT", DATA_DIR / "semantic" / "sentences.txt"))

# Default Hugging Face repository ID for sentence embeddings
SEMANTIC_MODEL_PATH = os.getenv(
    "VSL_SEMANTIC_MODEL",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
)

# ====================================================================
# WEB APPLICATION & OUTPUT PATHS
# ====================================================================
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = Path(os.getenv("VSL_RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Web Server Settings
HOST = os.getenv("VSL_HOST", "0.0.0.0")
PORT = int(os.getenv("VSL_PORT", 8000))
