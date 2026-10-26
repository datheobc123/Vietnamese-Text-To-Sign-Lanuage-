# Vietnamese Text-to-Sign Language (VSL) Translator 🇻🇳 🤟

An end-to-end framework for translating natural Vietnamese text into **Vietnamese Sign Language (VSL)**, generating smooth skeletal pose keypoints, and rendering MP4 sign language videos.

This project provides both an interactive **FastAPI Web Application** with real-time video playback and a **Command-Line Interface (CLI)**. The pipeline features a neural Text-to-Gloss translation model (**ViT5**), a hybrid sign retrieval strategy (**Greedy Matching + PhoBERT FAISS Semantic Search**), and a phonetic fingerspelling mechanism (**NFD Unicode Decomposition**) for out-of-vocabulary (OOV) terms.

---

## 📌 Pipeline Architecture

```
                    ┌─────────────────────────────────┐
                    │     Input Vietnamese Text       │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │        Text-to-Gloss (Fine-tuned ViT5)           │
            │   Translates natural text to VSL gloss grammar   │
            └────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                  Sign Token Retrieval Engine                     │
    │                                                                  │
    │  1. Greedy Exact Match     : Longest dictionary n-gram matching  │
    │  2. Semantic Search        : PhoBERT + FAISS vector similarity   │
    │  3. NFD Fingerspelling     : Letter + Tone mark decomposition    │
    └────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │          Pose Stitching & Normalization          │
            │  - Hold Frames: Natural pauses between gestures  │
            │  - Smooth Transitions: Linear pose interpolation │
            │  - Landmark Normalization (pose-format library)  │
            └────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
            ┌──────────────────────────────────────────────────┐
            │             Video Rendering Pipeline             │
            │   PoseVisualizerCustom -> Output MP4/WebM Video  │
            └──────────────────────────────────────────────────┘
```

---

## 📂 Project Organization

All paths inside the repository are relative:

```
upload_github/
├── app.py                      # FastAPI Web Application & REST API endpoint (/translate)
├── main.py                     # Command-line interface runner (CLI)
├── config.py                   # Centralized configuration (relative paths & env vars)
├── grammar_convert.py          # ViT5 Seq2Seq Text-to-Gloss inference wrapper
├── text2sign.py                # Core Translation Engine (Retrieval, Pose stitching, Rendering)
├── improve_render.py           # Standardized 640x360 skeleton visualizer
├── imrpove_render.py           # Backward-compatibility alias
├── interpolation.py            # Pose frame interpolation utility
├── test.py                     # Batch testing script on sample sentences
├── new_test.py                 # Logging and diagnostic test script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules for caches, outputs, and heavy weights
├── .gitattributes              # Git LFS tracking configuration
├── README.md                   # Project documentation
├── data/
│   ├── dictionary/
│   │   └── test.json           # 4,390-word Vietnamese to Sign Video ID dictionary
│   ├── semantic/
│   │   ├── phrases.index       # Precomputed FAISS index for semantic search
│   │   └── sentences.txt       # Canonical phrase list matching the FAISS index
│   ├── sample_poses/           # Sample .pose files for instant out-of-the-box testing
│   ├── poses/                  # Destination directory for the full .pose dataset
│   ├── Corpus-Vie-VSL-10k/     # 10,000-sentence parallel corpus (CSV & XLSX)
│   ├── Corpus-Vie-VSL-1k3/     # 1,300-sentence benchmark test set with predictions
│   └── vietnamese-stopwords.txt
├── models/
│   └── vit5_sign_grammar/      # Directory for fine-tuned ViT5 checkpoint weights
├── static/                     # Web UI static assets
│   ├── index.html              # Frontend web interface (Greedy & Hybrid mode toggling)
│   ├── logo.png                # Branding logo
│   └── image1.png              # UI illustration asset
├── text2gloss/                 # Text-to-Gloss model training and evaluation
│   ├── train/
│   │   └── train.py            # ViT5 fine-tuning script using Seq2SeqTrainer
│   ├── evaluate.py             # Evaluation script for BLEU, ROUGE, and WER metrics
│   └── weight/                 # Placeholder for training checkpoints
└── results/                    # Output directory for rendered videos and poses
```

---

## ⚙️ Installation

### 1. Prerequisites
* Python 3.9 - 3.11
* FFmpeg (required by `opencv` and `vidgear` for MP4 video encoding)
* NVIDIA GPU with CUDA support (recommended for model inference; CPU execution is fully supported)

Install FFmpeg on Debian / Ubuntu:
```bash
sudo apt update && sudo apt install -y ffmpeg
```

### 2. Set Up Virtual Environment

Using standard `venv`:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or using [`uv`](https://github.com/astral-sh/uv) for fast installation:
```bash
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

---

## 📦 Data & Pretrained Models Setup

The repository utilizes relative paths by default via `config.py`. You can run out of the box using the included sample assets, or configure full dataset paths via environment variables.

### 1. ViT5 Text-to-Gloss Model
Place your fine-tuned ViT5 model checkpoint files inside `models/vit5_sign_grammar/`:
```
models/vit5_sign_grammar/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── spiece.model
├── tokenizer_config.json
└── tokenizer.json
```
*Optional*: If your checkpoint is stored in a custom path, specify it via:
```bash
export VSL_MODEL_DIR="path/to/vit5_checkpoint"
```

### 2. Pose Keypoint Dataset
* The repository includes common gestures (greetings, alphabet letters for fingerspelling) inside `data/sample_poses/` for immediate verification.
* To use the full sign dictionary (8,700+ `.pose` files), extract them into `data/poses/` or point to the directory:
```bash
export VSL_POSE_DIR="path/to/full_pose_dataset"
```

### 3. Semantic Search Resources
* The dictionary mapping (`data/dictionary/test.json`) and FAISS index (`data/semantic/phrases.index`) are already included.
* The embedding model (`VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`) will automatically download from Hugging Face Hub on first run (or point `VSL_SEMANTIC_MODEL` to a local directory).

---

## 🚀 Usage

### 1. Running the Web Application (FastAPI)

Start the server:
```bash
python app.py
```
Or launch using `uvicorn`:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to: **`http://localhost:8000`**
1. Enter Vietnamese text (e.g., *"Tôi muốn đi siêu thị"*).
2. Select the search algorithm:
   * **Greedy Search**: Fast exact matching prioritizing longest n-gram tokens.
   * **Hybrid Search**: Combines exact matching with PhoBERT + FAISS semantic similarity.
3. Click **"Translate to Vietnamese Sign Language"** to view the VSL Gloss grammar structure and watch the rendered sign language video.

---

### 2. Command-Line Interface (CLI)

Use `main.py` for command-line inference:

```bash
# Basic translation and rendering
python main.py -t "Chào bạn" -v results/output_chao.mp4

# Hybrid semantic search with custom pose output
python main.py -t "Tôi yêu bạn nhiều vô cùng" -s hybrid -v results/love.mp4 -p results/love.pose
```

Supported arguments:
* `-t, --text`: Input Vietnamese sentence (default: `"Tôi yêu bạn nhiều vô cùng"`).
* `-v, --video`: Path for the output MP4 video (default: `output.mp4`).
* `-p, --pose`: Path for the output `.pose` file (default: `output.pose`).
* `-s, --search-mode`: Search strategy (`greedy` or `hybrid`).

---

### 3. Batch Testing

To verify video rendering across a suite of test sentences:
```bash
python test.py
```
Generated video files are saved to `results/batch_test/`.

---

## 🔬 Text-to-Gloss Training & Evaluation

### 1. Fine-tuning ViT5
Train the Text-to-Gloss model on the 10K parallel corpus using Hugging Face's `Seq2SeqTrainer`:
```bash
python text2gloss/train/train.py
```

### 2. Quantitative Evaluation
Compute BLEU-1 to BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, and WER scores:
```bash
python text2gloss/evaluate.py data/Corpus-Vie-VSL-1k3/new_1k3_test_with_predict.csv
```

---

## 🔧 Environment Configuration Reference

All settings can be configured via environment variables:

| Variable | Default (Relative Path) | Description |
| :--- | :--- | :--- |
| `VSL_DATA_DIR` | `data` | Base directory for data assets |
| `VSL_MAPPING_FILE` | `data/dictionary/test.json` | Token-to-label dictionary mapping file |
| `VSL_POSE_DIR` | `data/poses` (fallback: `data/sample_poses`) | Directory containing `.pose` landmark files |
| `VSL_MODEL_DIR` | `models/vit5_sign_grammar` | Directory containing fine-tuned ViT5 weights |
| `VSL_FAISS_INDEX` | `data/semantic/phrases.index` | FAISS vector index for semantic retrieval |
| `VSL_SENTENCES_TXT` | `data/semantic/sentences.txt` | Text file of canonical phrases for FAISS |
| `VSL_SEMANTIC_MODEL` | `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` | SentenceTransformer model name or path |
| `VSL_RESULTS_DIR` | `results` | Output directory for rendered videos |
| `VSL_HOST` | `0.0.0.0` | Web server host address |
| `VSL_PORT` | `8000` | Web server port |

---

## 📜 Citation

If you use this codebase, dataset, or pipeline in your research, please cite:

```bibtex
@misc{vsl_translator_2026,
  author = {Quan, N. H. and Contributors},
  title = {Vietnamese Text-to-Sign Language (VSL) Translation System},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/datheobc123/VSL_UIT}}
}
```
