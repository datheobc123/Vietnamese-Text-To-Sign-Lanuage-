# =========================================
# evaluate_csv.py
# =========================================

import pandas as pd
import numpy as np
import re
import unicodedata
from jiwer import wer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys

# ==============================
# Text normalization
# ==============================
def normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[.,!?;:]", "", text)  # remove punctuation
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])  # remove accents
    text = re.sub(r"\s+", " ", text).strip()
    return text



# ==============================
# Compute BLEU and ROUGE
# ==============================
def compute_bleu_rouge(refs, preds):
    # BLEU
    smoother = SmoothingFunction().method1
    bleu_list = []
    for ref, pred in zip(refs, preds):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
        bleu_list.append(bleu)
    
    # ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    for ref, pred in zip(refs, preds):
        scores = rouge.score(ref, pred)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

    return np.mean(bleu_list), np.mean(rouge1_list), np.mean(rouge2_list), np.mean(rougeL_list)

# ==============================
# Main evaluation function
# ==============================
def evaluate_csv(csv_path, ref_col="output", pred_col="predict", print_scores=True):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    references = df[ref_col].astype(str).tolist()
    predictions = df[pred_col].astype(str).tolist()

    # Normalize
    refs_norm = [normalize_text(r) for r in references]
    preds_norm = [normalize_text(p) for p in predictions]

    # Compute WER (corpus-level)
    wer_score = wer(refs_norm, preds_norm)

    # Compute BLEU and ROUGE
    bleu, rouge1, rouge2, rougeL = compute_bleu_rouge(refs_norm, preds_norm)

    if print_scores:
        # print("===== Evaluation Results =====")
        # print(f"Corpus-level WER : {wer_score:.4f}")
        # print(f"BLEU             : {bleu*100:.2f}%")
        # print(f"ROUGE-1          : {rouge1*100:.2f}%")
        # print(f"ROUGE-2          : {rouge2*100:.2f}%")
        # print(f"ROUGE-L          : {rougeL*100:.2f}%")
        
        print("===== Evaluation Results =====")
        print(f"{wer_score*100:.2f}%")
        print(f"{bleu*100:.2f}%")
        print(f"{rouge1*100:.2f}%")
        print(f"{rouge2*100:.2f}%")
        print(f"{rougeL*100:.2f}%")

    return {
        "WER": wer_score,
        "BLEU": bleu,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL
    }

# ==============================
# Run from command line
# ==============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_csv.py <path_to_csv> [ref_column] [pred_column]")
        sys.exit(1)

    csv_path = sys.argv[1]
    ref_col = sys.argv[2] if len(sys.argv) > 2 else "output"
    pred_col = sys.argv[3] if len(sys.argv) > 3 else "predict"

    evaluate_csv(csv_path, ref_col, pred_col)
# python evaluate_csv.py path/to/your_file.csv ref_column_name pred_column_name
