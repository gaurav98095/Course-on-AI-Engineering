"""Retrieve + generate: the whole multimodal RAG in one file.

Run:  python rag.py "Why does an aircraft stall at the critical angle of attack?"
      python rag.py "What does this instrument do?" --image path/to/photo.jpg
"""

import argparse
import json
from pathlib import Path

import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

GENERATOR = "Qwen/Qwen3-VL-8B-Instruct"
TEXT_EMBEDDER = "BAAI/bge-small-en-v1.5"
CLIP_EMBEDDER = "google/siglip2-base-patch16-224"
CORPUS = Path("corpus")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM = (
    "You are a technical assistant for the attached manual. Answer only from the "
    "provided excerpts and figures. Cite pages like [ch4 p.12]. If the context is "
    "not enough, say so."
)


class Retriever:
    def __init__(self) -> None:
        meta = json.loads((CORPUS / "chunks.json").read_text())
        self.texts, self.images = meta["texts"], meta["images"]
        self.text_index = faiss.read_index(str(CORPUS / "text.faiss"))
        self.image_index = faiss.read_index(str(CORPUS / "image.faiss"))
        self.text_model = SentenceTransformer(TEXT_EMBEDDER, device=DEVICE)
        self.clip = AutoModel.from_pretrained(CLIP_EMBEDDER, torch_dtype=torch.float16).to(DEVICE)
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_EMBEDDER)

    @torch.inference_mode()
    def __call__(self, question: str, image: Image.Image | None = None,
                 k_text: int = 4, k_img: int = 2) -> tuple[list[dict], list[dict]]:
        # text -> text index (BGE lives in its own space)
        q = self.text_model.encode([question], normalize_embeddings=True)
        _, idx = self.text_index.search(q, k_text)
        hits_t = [self.texts[i] for i in idx[0]]

        # text (or image) -> image index (SigLIP shared space)
        if image is not None:
            inputs = self.clip_processor(images=[image], return_tensors="pt").to(DEVICE)
            qv = self.clip.get_image_features(**inputs)
        else:
            # SigLIP's text tower is trained on <=64-token captions: padding is mandatory
            inputs = self.clip_processor(
                text=[question], padding="max_length", max_length=64, return_tensors="pt"
            ).to(DEVICE)
            qv = self.clip.get_text_features(**inputs)
        qv = qv.pooler_output if hasattr(qv, "pooler_output") else qv
        qv = torch.nn.functional.normalize(qv, dim=-1).float().cpu().numpy()
        _, idx = self.image_index.search(qv, k_img)
        hits_i = [self.images[i] for i in idx[0]]
        return hits_t, hits_i


class Generator:
    def __init__(self, quantized_path: str | None = None) -> None:
        if quantized_path is None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                GENERATOR, torch_dtype="auto", device_map="auto"
            )
        else:
            # GPTQModel quantizes only the text layers (Lecture 08) -- the
            # vision tower and processor are unchanged, so we still load the
            # stock processor below and only swap the generator's weights.
            from gptqmodel import GPTQModel
            self.model = GPTQModel.load(quantized_path)
        self.processor = AutoProcessor.from_pretrained(GENERATOR)

    @torch.inference_mode()
    def __call__(self, question: str, hits_t: list[dict], hits_i: list[dict],
                 user_image: Image.Image | None = None, max_new_tokens: int = 400):
        content = []
        for m in hits_i:  # retrieved figures go in as real images, not descriptions
            content.append({"type": "image", "image": str(CORPUS / "images" / m["file"])})
        if user_image is not None:
            content.append({"type": "image", "image": user_image})
        excerpts = "\n\n".join(f"[{t['doc']} p.{t['page']}] {t['text']}" for t in hits_t)
        content.append({"type": "text", "text": f"Manual excerpts:\n{excerpts}\n\nQuestion: {question}"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return answer, inputs["input_ids"].shape[1], new_tokens.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--image", help="optional query image")
    args = ap.parse_args()

    retrieve, generate = Retriever(), Generator()
    user_image = Image.open(args.image).convert("RGB") if args.image else None

    hits_t, hits_i = retrieve(args.question, user_image)
    answer, n_in, n_out = generate(args.question, hits_t, hits_i, user_image)

    print("\n--- retrieved:")
    for t in hits_t:
        print(f"  [{t['doc']} p.{t['page']}] {t['text'][:80]}...")
    for m in hits_i:
        print(f"  [figure] {m['file']}")
    print(f"\n--- answer ({n_in} prompt tokens -> {n_out} new tokens):\n{answer}")


if __name__ == "__main__":
    main()
