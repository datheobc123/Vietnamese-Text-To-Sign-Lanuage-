from typing import Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import MODEL_DIR as DEFAULT_MODEL_DIR


class GrammarConvert:
    """
    Chuyển đổi văn bản tiếng Việt sang cấu trúc ngữ pháp Ngôn ngữ Ký hiệu (VSL Gloss)
    bằng mô hình ViT5 Seq2Seq được tinh chỉnh.
    """
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = str(model_dir) if model_dir else str(DEFAULT_MODEL_DIR)

        print(f"⏳ Đang nạp mô hình GrammarConvert từ: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Nạp mô hình GrammarConvert thành công trên thiết bị: {self.device}")

    def convert(self, text: str) -> str:
        """Dịch câu tiếng Việt đầu vào sang chuỗi VSL Gloss."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=5,
                do_sample=False,
                temperature=0.8,
                top_p=0.9,
                early_stopping=True
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return decoded.strip()


if __name__ == "__main__":
    test_sentence = "Tôi muốn đi siêu thị ."
    converter = GrammarConvert()
    vsl_result = converter.convert(test_sentence)
    print(f"Tiếng Việt: {test_sentence}")
    print(f"VSL Gloss : {vsl_result}")
