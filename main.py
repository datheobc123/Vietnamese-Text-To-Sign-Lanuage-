"""
CLI Tool: Chuyển đổi văn bản Tiếng Việt sang Video Ngôn ngữ Ký hiệu (VSL).
Ví dụ sử dụng:
    python main.py -t "Chào bạn" -v output.mp4
"""
import argparse
import sys
from pathlib import Path

from config import (
    MAPPING_FILE,
    POSE_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    SEMANTIC_MODEL_PATH,
    FAISS_INDEX_PATH,
    SENTENCES_TXT_PATH
)
from grammar_convert import GrammarConvert
from text2sign import Text2SignCoverter

sys.stdin.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Text-to-Sign CLI Pipeline")
    parser.add_argument("--text", "-t", type=str, default="Tôi yêu bạn nhiều vô cùng", help="Câu tiếng Việt cần dịch")
    parser.add_argument("--video", "-v", type=str, default="output.mp4", help="Đường dẫn tệp video MP4 đầu ra")
    parser.add_argument("--pose", "-p", type=str, default="output.pose", help="Đường dẫn tệp pose đầu ra")
    parser.add_argument("--search-mode", "-s", type=str, default="greedy", choices=["greedy", "hybrid"], help="Chế độ tìm kiếm ký hiệu")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Đường dẫn mô hình ViT5")
    parser.add_argument("--pose-dir", type=str, default=str(POSE_DIR), help="Thư mục chứa các tệp .pose")
    parser.add_argument("--mapping-file", type=str, default=str(MAPPING_FILE), help="Tệp JSON từ điển ánh xạ VSL")

    args = parser.parse_args()

    print("=" * 60)
    print("🎬 VSL Text-to-Sign CLI Runner")
    print(f"📝 Câu đầu vào : {args.text}")
    print(f"🔍 Chế độ tìm  : {args.search_mode}")
    print("=" * 60)

    # 1. Khởi tạo bộ chuyển đổi ngữ pháp
    grammar_converter = GrammarConvert(model_dir=args.model_dir)
    sign_sentence = grammar_converter.convert(args.text)
    print(f"✨ Ngữ pháp VSL (Gloss): {sign_sentence}")

    # 2. Khởi tạo Text2SignConverter
    translator = Text2SignCoverter(
        mapping_file=args.mapping_file,
        pose_dir=args.pose_dir,
        model_dir=args.model_dir,
        semantic_model_path=str(SEMANTIC_MODEL_PATH) if SEMANTIC_MODEL_PATH else None,
        faiss_index_path=str(FAISS_INDEX_PATH) if Path(FAISS_INDEX_PATH).exists() else None,
        sentences_txt_path=str(SENTENCES_TXT_PATH) if Path(SENTENCES_TXT_PATH).exists() else None
    )

    # 3. Thực hiện chuyển đổi và kết xuất video
    print("🎥 Bắt đầu ghép khung xương và xuất video...")
    pose = translator.process(
        text=sign_sentence,
        output_pose=args.pose,
        output_video=args.video,
        search_type=args.search_mode
    )

    if pose:
        print("=" * 60)
        print(f"🎉 Hoàn thành thành công!")
        print(f"💾 File Pose : {args.pose}")
        print(f"🎬 File Video: {args.video}")
        print("=" * 60)
    else:
        print("❌ Không thể tạo video ký hiệu.")


if __name__ == "__main__":
    main()
