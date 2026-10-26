import unicodedata
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pose_format import Pose
from improve_render import PoseVisualizerCustom
from grammar_convert import GrammarConvert
from config import MAPPING_FILE, POSE_DIR, MODEL_DIR, RESULTS_DIR

CSV_OUTPUT = RESULTS_DIR / "text2sign_log.csv"

TONE_MARKS = {
    '\u0300': 'dấu huyền',
    '\u0301': 'dấu sắc',
    '\u0303': 'dấu ngã',
    '\u0309': 'dấu hỏi',
    '\u0323': 'dấu nặng'
}


def tach_chu_tieng_viet(word):
    decomposed = unicodedata.normalize('NFD', word)
    chars = []
    tone = None

    for ch in decomposed:
        if ch in TONE_MARKS:
            tone = TONE_MARKS[ch]
        elif unicodedata.category(ch) != 'Mn':
            chars.append(ch.lower())

    return chars, tone


def get_video_ids_and_log(text_obj, mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        vsl_mapping = json.load(f)

    text = text_obj.text.lower().strip()
    words = text.split()

    token_to_label = {}
    for group in vsl_mapping:
        for item in group.get('mapping', []):
            label = item.get('video_id', item.get('label'))
            if not label:
                continue
            for token in item.get('token', {}).get('vi', []):
                token = token.lower().strip()
                token_to_label.setdefault(token, []).append(label)

    n = len(words)
    i = 0
    video_ids = []
    logs = []

    while i < n:
        best_match_label = None
        match_length = 0

        for length in range(n - i, 0, -1):
            token_to_check = " ".join(words[i:i + length])
            if token_to_check in token_to_label:
                labels = token_to_label[token_to_check]
                best_match_label = next((l for l in labels if l.endswith('N')), labels[0])
                match_length = length
                logs.append({
                    "original_part": token_to_check,
                    "matched_token": token_to_check,
                    "label": best_match_label,
                    "type": "exact_phrase"
                })
                break

        if best_match_label:
            video_ids.append(best_match_label)
            i += match_length
        else:
            word = words[i]
            letters, tone = tach_chu_tieng_viet(word)

            for ch in letters:
                if ch in token_to_label:
                    labels = token_to_label[ch]
                    lbl = next((l for l in labels if l.endswith('N')), labels[0])
                    video_ids.append(lbl)
                    logs.append({
                        "original_part": word,
                        "matched_token": ch,
                        "label": lbl,
                        "type": "spelling_letter"
                    })

            if tone and tone in token_to_label:
                labels = token_to_label[tone]
                lbl = next((l for l in labels if l.endswith('N')), labels[0])
                video_ids.append(lbl)
                logs.append({
                    "original_part": word,
                    "matched_token": tone,
                    "label": lbl,
                    "type": "spelling_tone"
                })

            i += 1

    return video_ids, logs


def save_log_csv(sentence, grammar_text, process_logs, video_ids, output_file=CSV_OUTPUT):
    rows = []
    for log in process_logs:
        rows.append({
            "original_sentence": sentence,
            "grammar_text": grammar_text,
            "original_part": log["original_part"],
            "matched_token": log["matched_token"],
            "label": log["label"],
            "type": log["type"]
        })

    df = pd.DataFrame(rows)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if Path(output_file).exists():
        df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Log appended to {output_file}")


if __name__ == "__main__":
    single_sentence = "Tôi có ba lô đất"
    class DummyText:
        text = single_sentence

    video_ids, process_logs = get_video_ids_and_log(DummyText, MAPPING_FILE)
    print("Video IDs:", video_ids)
    print("Process Logs:", process_logs)
