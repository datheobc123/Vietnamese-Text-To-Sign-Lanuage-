import unicodedata
import json
from pathlib import Path
import numpy as np
import re
from typing import List, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info, normalize_pose_size

try:
    from improve_render import PoseVisualizerCustom
except ImportError:
    PoseVisualizerCustom = None

from config import (
    MAPPING_FILE as DEFAULT_MAPPING_FILE,
    POSE_DIR as DEFAULT_POSE_DIR,
    MODEL_DIR as DEFAULT_MODEL_DIR,
    SEMANTIC_MODEL_PATH as DEFAULT_SEMANTIC_MODEL_PATH,
    FAISS_INDEX_PATH as DEFAULT_FAISS_INDEX_PATH,
    SENTENCES_TXT_PATH as DEFAULT_SENTENCES_TXT_PATH
)

TONE_MARKS = {
    '\u0300': 'dấu huyền',
    '\u0301': 'dấu sắc',
    '\u0303': 'dấu ngã',
    '\u0309': 'dấu chấm hỏi',
    '\u0323': 'dấu nặng'  
}

# ====================================================================
# HÀM HỖ TRỢ XỬ LÝ KHUNG HÌNH (POSE TRANSITION)
# ====================================================================

def smooth_transition(frame_a: np.ndarray, frame_b: np.ndarray, steps: int = 6) -> np.ndarray:
    """Nội suy tuyến tính giữa 2 frame để tạo hiệu ứng chuyển động mượt mà."""
    frames = []
    for i in range(1, steps + 1):
        alpha = i / (steps + 1)
        inter = (1 - alpha) * frame_a + alpha * frame_b
        frames.append(inter)
    return np.stack(frames)


def hold_last_frame(data: np.ndarray, conf: np.ndarray, hold: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Lặp lại frame cuối cùng để tạo khoảng dừng tự nhiên giữa các từ ký hiệu."""
    last_frame = data[-1:]
    last_conf = conf[-1:]
    hold_data = np.repeat(last_frame, hold, axis=0)
    hold_conf = np.repeat(last_conf, hold, axis=0)
    return hold_data, hold_conf


# ====================================================================
# XỬ LÝ VĂN BẢN VÀ ĐÁNH VẦN TIẾNG VIỆT
# ====================================================================

def split_words_and_punctuation(text: str) -> List[str]:
    """Tách từ và dấu câu thành danh sách token."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def tokenize(word: str) -> Tuple[List[str], Optional[str]]:
    """
    Tách từ thành danh sách chữ cái và dấu thanh dựa trên phân rã Unicode NFD.
    Giữ lại các ký tự có dấu mũ/râu (ô, â, ê, ă, ơ, ư) và tách riêng dấu thanh.
    """
    decomposed = unicodedata.normalize('NFD', word)
    chars = []
    tone = None
    for c in decomposed:
        if c in TONE_MARKS:
            tone = TONE_MARKS[c]
        elif unicodedata.category(c) != 'Mn':
            chars.append(c.lower())
        elif chars:
            chars[-1] += c

    chars = [unicodedata.normalize('NFC', ch) for ch in chars]
    return chars, tone


# ====================================================================
# CLASS CHÍNH: Text2SignCoverter / Text2SignConverter
# ====================================================================

class Text2SignCoverter:
    def __init__(
        self,
        mapping_file: Optional[str] = None,
        pose_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        semantic_model_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        sentences_txt_path: Optional[str] = None,
        semantic_threshold: float = 0.90
    ):
        """
        Text -> Pose -> Video Engine hỗ trợ cả Greedy Search và Semantic Hybrid Search.
        """
        self.mapping_file = Path(mapping_file) if mapping_file else Path(DEFAULT_MAPPING_FILE)
        self.pose_dir = Path(pose_dir) if pose_dir else Path(DEFAULT_POSE_DIR)
        self.model_dir = Path(model_dir) if model_dir else Path(DEFAULT_MODEL_DIR)
        self.semantic_threshold = semantic_threshold

        # 1. Nạp từ điển ánh xạ VSL (token -> video_id / label)
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Không tìm thấy mapping_file tại: {self.mapping_file}")

        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_raw = json.load(f)

        self.token_to_label = {}
        for group in mapping_raw:
            for item in group.get("mapping", []):
                label = item.get("video_id", item.get("label"))
                if not label:
                    continue
                for token in item.get("token", {}).get("vi", []):
                    token = token.lower().strip()
                    self.token_to_label.setdefault(token, []).append(label)

        # 2. Khởi tạo mô hình Semantic Search & FAISS (nếu có tài nguyên)
        self.semantic_model = None
        self.faiss_index = None
        self.sentences = []

        sem_path = semantic_model_path or DEFAULT_SEMANTIC_MODEL_PATH
        index_path = faiss_index_path or DEFAULT_FAISS_INDEX_PATH
        sent_path = sentences_txt_path or DEFAULT_SENTENCES_TXT_PATH

        if sem_path and index_path and sent_path:
            idx_file = Path(index_path)
            s_file = Path(sent_path)
            if idx_file.exists() and s_file.exists():
                try:
                    print("⏳ Đang nạp SentenceTransformer và FAISS index cho Hybrid Search...")
                    self.semantic_model = SentenceTransformer(str(sem_path))
                    self.semantic_model.max_seq_length = 2048
                    self.faiss_index = faiss.read_index(str(idx_file))
                    with open(s_file, "r", encoding="utf-8") as f:
                        self.sentences = [line.strip() for line in f if line.strip()]
                    print("✅ Nạp Semantic Model & FAISS Index thành công!")
                except Exception as e:
                    print(f"⚠️ Không thể khởi tạo Semantic Model ({e}). Sẽ chỉ dùng Greedy Search.")

    def semantic_search_phrase(self, sub_phrase: str) -> Tuple[Optional[str], float]:
        """Tìm kiếm câu/cụm tương đương về mặt ngữ nghĩa trong FAISS index."""
        if not self.semantic_model or not self.faiss_index or not self.sentences:
            return None, 0.0

        query_emb = self.semantic_model.encode(
            [sub_phrase], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.faiss_index.search(query_emb.astype("float32"), 1)
        score = float(scores[0][0])
        matched_idx = int(indices[0][0])
        if 0 <= matched_idx < len(self.sentences):
            return self.sentences[matched_idx], score
        return None, 0.0

    def greedy_match_phrase(self, phrase: str) -> Tuple[Optional[str], Optional[str]]:
        """Khớp cụm từ dài nhất trong từ điển."""
        tokens = split_words_and_punctuation(phrase.lower())
        n = len(tokens)
        for length in range(n, 0, -1):
            for start in range(0, n - length + 1):
                candidate = " ".join(tokens[start:start + length])
                if candidate in self.token_to_label:
                    labels = self.token_to_label[candidate]
                    best_label = next((l for l in labels if l.endswith("N")), labels[0])
                    return candidate, best_label
        return None, None

    def get_video_ids(self, text: str, search_type: str = "greedy") -> List[str]:
        """
        Tìm danh sách video ID tương ứng với từng token trong văn bản.
        Hỗ trợ:
          - Greedy Exact Match
          - Hybrid Semantic Search (kết hợp FAISS)
          - Fingerspelling (đánh vần ký tự + dấu thanh) khi từ ngoài từ điển
        """
        text = text.lower().strip()
        tokens = split_words_and_punctuation(text)
        n = len(tokens)
        video_ids = []
        i = 0

        while i < n:
            best_match_label = None
            match_length = 0

            # 1. Greedy Exact Match
            for length in range(n - i, 0, -1):
                candidate = " ".join(tokens[i:i + length])
                if candidate in self.token_to_label:
                    labels = self.token_to_label[candidate]
                    best_match_label = next((l for l in labels if l.endswith('N')), labels[0])
                    match_length = length
                    break

            if best_match_label:
                video_ids.append(best_match_label)
                i += match_length
                continue

            token = tokens[i]

            # Bỏ qua dấu câu nếu không có ký hiệu riêng
            if re.match(r"[^\w\s]", token):
                i += 1
                continue

            # 2. Semantic Search (nếu search_type == 'hybrid')
            semantic_found = False
            if search_type == "hybrid" and self.semantic_model:
                for length in range(n - i, 0, -1):
                    sub_phrase = " ".join(tokens[i:i + length])
                    matched_text, score = self.semantic_search_phrase(sub_phrase)
                    if matched_text and score >= self.semantic_threshold:
                        cand, label = self.greedy_match_phrase(matched_text)
                        if label:
                            video_ids.append(label)
                            i += length
                            semantic_found = True
                            break

            if semantic_found:
                continue

            # 3. Fallback: Đánh vần từng ký tự và dấu thanh
            chars, tone = tokenize(token)
            for ch in chars:
                if ch in self.token_to_label:
                    labels = self.token_to_label[ch]
                    lbl = next((l for l in labels if l.endswith('N')), labels[0])
                    video_ids.append(lbl)

            if tone and tone in self.token_to_label:
                labels = self.token_to_label[tone]
                lbl = next((l for l in labels if l.endswith('N')), labels[0])
                video_ids.append(lbl)

            i += 1

        return video_ids

    def merge_pose_files(self, video_ids: List[str], output_path: str = "output.pose", normalize: bool = True) -> Optional[Pose]:
        """Đọc và ghép các file pose thành một chuỗi mượt mà."""
        poses = []
        for vid in video_ids:
            pose_file = self.pose_dir / f"{vid}.pose"
            if not pose_file.exists():
                print(f"⚠️ Không tìm thấy file pose: {pose_file.name}")
                continue
            with open(pose_file, "rb") as f:
                poses.append(Pose.read(f.read()))

        if not poses:
            print("❌ Không có file pose nào được nạp.")
            return None

        all_poses_data = []
        all_poses_conf = []
        base_pose = poses[0]

        for i, p in enumerate(poses):
            data = p.body.data
            conf = p.body.confidence

            all_poses_data.append(data)
            all_poses_conf.append(conf)

            # Thêm khung hình giữ nhịp (Hold)
            h_data, h_conf = hold_last_frame(data, conf, hold=5)
            all_poses_data.append(h_data)
            all_poses_conf.append(h_conf)

            # Thêm khung chuyển tiếp mượt mà giữa 2 từ
            if i < len(poses) - 1:
                next_first_frame = poses[i + 1].body.data[0]
                smooth_data = smooth_transition(data[-1], next_first_frame, steps=6)
                all_poses_data.append(smooth_data)
                smooth_conf = np.ones((6, *conf.shape[1:]))
                all_poses_conf.append(smooth_conf)

        merged_data = np.concatenate(all_poses_data, axis=0)
        merged_conf = np.concatenate(all_poses_conf, axis=0)

        merged_pose = Pose(base_pose.header, base_pose.body)
        merged_pose.body.data = merged_data
        merged_pose.body.confidence = merged_conf

        if normalize:
            merged_pose = merged_pose.normalize(pose_normalization_info(merged_pose.header))
            normalize_pose_size(merged_pose)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            merged_pose.write(f)

        return merged_pose

    def render_video(self, pose: Pose, output_path: str, out_w: int = 640, out_h: int = 360):
        """Vẽ khung xương và xuất file MP4."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if PoseVisualizerCustom is not None:
            visualizer = PoseVisualizerCustom(pose, out_w=out_w, out_h=out_h)
            visualizer.save_video(output_path, visualizer.draw())
        else:
            visualizer = PoseVisualizer(pose)
            visualizer.save_video(output_path, visualizer.draw())

    def process(self, text: str, output_pose: str, output_video: str, search_type: str = "greedy") -> Optional[Pose]:
        """Thực hiện quy trình trọn vẹn: Text -> Video IDs -> Merged Pose -> MP4 Video."""
        video_ids = self.get_video_ids(text, search_type=search_type)
        if not video_ids:
            print("❌ Không tìm thấy video id nào cho văn bản này.")
            return None

        pose = self.merge_pose_files(video_ids, output_path=output_pose, normalize=True)
        if pose:
            self.render_video(pose, output_path=output_video)
        return pose


# Alias hỗ trợ cả 2 cách viết tên lớp
Text2SignConverter = Text2SignCoverter
