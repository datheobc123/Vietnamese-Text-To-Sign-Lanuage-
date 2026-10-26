"""
Batch Testing Script for VSL Text-to-Sign Pipeline.
"""
import os
from pathlib import Path
from text2sign import Text2SignCoverter
from config import MAPPING_FILE, POSE_DIR, MODEL_DIR, RESULTS_DIR

# 1. Khởi tạo converter với cấu hình tự động từ config
converter = Text2SignCoverter(
    mapping_file=str(MAPPING_FILE),
    pose_dir=str(POSE_DIR),
    model_dir=str(MODEL_DIR)
)

# 2. Danh sách các câu tiếng Việt mẫu để kiểm thử
sentences = [
    "Địa chỉ nhà tôi ở Thành Phố Hồ Chí Minh",
    "Ngày ngôn ngữ ký hiệu quốc tế 23 9",
    "Nước ép trái cây, muốn không",
    "Môn tin học và công nghệ, thích thú",
    "Màu đỏ , màu cam không thích",
    "tôi yêu bạn nhiều vô cùng",
    "các bạn cần khởi động trước khi tập thể dục",
    "bạn tên là gì ?",
    "chào buổi sáng",
    "chào buổi tối",
    "chào mừng ngày nhà giáo việt nam"
]

output_dir = RESULTS_DIR / "batch_test"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"🎬 Bắt đầu render {len(sentences)} câu mẫu...")
print(f"📁 Thư mục lưu kết quả: {output_dir}")
print("=" * 60)

for index, sentence in enumerate(sentences):
    file_name = f"video_{index + 1:02d}"
    out_pose = str(output_dir / f"{file_name}.pose")
    out_video = str(output_dir / f"{file_name}.mp4")

    print(f"\n[{index + 1}/{len(sentences)}] Đang xử lý: '{sentence}'")
    converter.process(
        text=sentence,
        output_pose=out_pose,
        output_video=out_video
    )

print("\n🎉 Đã hoàn tất render tất cả các câu kiểm thử!")
