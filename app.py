"""
FastAPI Web Application for Vietnamese Text-to-Sign Language (VSL) Translator.
"""
from typing import Optional
from pathlib import Path
import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    MAPPING_FILE,
    POSE_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    STATIC_DIR,
    SEMANTIC_MODEL_PATH,
    FAISS_INDEX_PATH,
    SENTENCES_TXT_PATH,
    HOST,
    PORT
)
from grammar_convert import GrammarConvert
from text2sign import Text2SignCoverter

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="VSL Text-to-Sign Translator", description="Vietnamese Text to Sign Language Web Application")

# Biến toàn cục lưu trữ translator và grammar converter
vsl_translator: Optional[Text2SignCoverter] = None
grammar_converter: Optional[GrammarConvert] = None

# Đảm bảo thư mục static và results tồn tại
STATIC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files và results
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

templates = Jinja2Templates(directory=STATIC_DIR)


def clear_old_results():
    """Xóa các video và pose tạm thời trong thư mục results."""
    for file in RESULTS_DIR.iterdir():
        if file.is_file() and file.name != ".gitkeep":
            try:
                file.unlink()
            except Exception as e:
                print(f"⚠️ Không thể xoá tệp cũ {file.name}: {e}")


def initialize_translator() -> bool:
    """Khởi tạo mô hình chuyển đổi ngữ pháp và trình ánh xạ ký hiệu."""
    global vsl_translator, grammar_converter
    print("=" * 60)
    print("⏳ Khởi tạo VSL Text-to-Sign Translator...")
    print(f"📁 Mapping File   : {MAPPING_FILE}")
    print(f"📁 Pose Directory : {POSE_DIR}")
    print(f"📁 Model Directory: {MODEL_DIR}")
    print(f"📁 FAISS Index    : {FAISS_INDEX_PATH}")
    print("=" * 60)

    # 1. Khởi tạo GrammarConvert (ViT5)
    if not MODEL_DIR.exists():
        print(f"❌ Lỗi: Không tìm thấy thư mục mô hình MODEL_DIR tại: {MODEL_DIR}")
        return False

    try:
        grammar_converter = GrammarConvert(model_dir=str(MODEL_DIR))
        print("✅ GrammarConvert (ViT5) khởi tạo thành công.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo GrammarConvert: {e}")
        return False

    # 2. Kiểm tra tài nguyên từ điển và pose
    if not MAPPING_FILE.exists():
        print(f"❌ Lỗi: Không tìm thấy MAPPING_FILE tại: {MAPPING_FILE}")
        return False

    if not POSE_DIR.is_dir():
        print(f"❌ Lỗi: Không tìm thấy POSE_DIR tại: {POSE_DIR}")
        return False

    # 3. Khởi tạo Text2SignCoverter
    try:
        vsl_translator = Text2SignCoverter(
            mapping_file=str(MAPPING_FILE),
            pose_dir=str(POSE_DIR),
            model_dir=str(MODEL_DIR),
            semantic_model_path=str(SEMANTIC_MODEL_PATH) if SEMANTIC_MODEL_PATH else None,
            faiss_index_path=str(FAISS_INDEX_PATH) if FAISS_INDEX_PATH.exists() else None,
            sentences_txt_path=str(SENTENCES_TXT_PATH) if SENTENCES_TXT_PATH.exists() else None
        )
        print("✅ Khởi tạo Text2SignCoverter thành công.")
        return True
    except Exception as e:
        print(f"❌ Lỗi trong quá trình khởi tạo Text2SignCoverter: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Hook chạy khi server khởi động."""
    success = initialize_translator()
    if not success:
        print("⚠️ Cảnh báo: Mô hình chưa sẵn sàng. Vui lòng kiểm tra lại cấu hình đường dẫn.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Trang giao diện người dùng chính."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate")
async def translate_text_to_sign(
    text: str = Form(...),
    search_mode: str = Form("greedy")
):
    """
    Endpoint dịch câu tiếng Việt sang chuỗi ký hiệu VSL và kết xuất video.
    """
    if vsl_translator is None or grammar_converter is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Mô hình dịch chưa được nạp. Vui lòng kiểm tra log server."}
        )

    text = text.strip()
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": "Vui lòng nhập văn bản tiếng Việt."}
        )

    try:
        clear_old_results()

        # 1. Dịch văn bản tiếng Việt sang cấu trúc ngữ pháp VSL Gloss
        sign_sentence = grammar_converter.convert(text)
        print(f"🔁 Input: '{text}' -> VSL Grammar: '{sign_sentence}'")

        # 2. Đặt tên tệp xuất
        safe_id = sign_sentence.replace(" ", "_").lower()
        # Loại bỏ các ký tự không hợp lệ cho tên tệp
        safe_id = "".join(c for c in safe_id if c.isalnum() or c in ("_", "-"))[:60]
        output_pose_file = RESULTS_DIR / f"{safe_id}.pose"
        output_video_file = RESULTS_DIR / f"{safe_id}.mp4"

        # 3. Chuyển đổi VSL Gloss sang Pose và render video MP4
        vsl_translator.process(
            text=sign_sentence,
            output_pose=str(output_pose_file),
            output_video=str(output_video_file),
            search_type=search_mode
        )

        if not output_video_file.exists():
            raise FileNotFoundError("Không tạo được video ký hiệu cho câu này.")

        video_url = f"/results/{output_video_file.name}"

        return JSONResponse(content={
            "success": True,
            "original_text": text,
            "sign_grammar": sign_sentence,
            "video_id": safe_id,
            "video_url": video_url,
            "search_mode_used": search_mode
        })

    except Exception as e:
        print(f"❌ Lỗi trong /translate: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    print(f"🚀 Khởi chạy VSL Translator Web Server tại http://{HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
