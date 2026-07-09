"""Ingest: turn an illustrated PDF manual into two searchable FAISS indexes.

Corpus: FAA Pilot's Handbook of Aeronautical Knowledge (public domain).
The two chapters ship with the repo in data/ — drop your own PDFs in there
(or add URLs to PDF_URLS) and nothing else changes.

Run:  python ingest.py
Out:  corpus/  (chunks.json, images/, text.faiss, image.faiss)
"""

import io
import json
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

PDF_URLS = {
    "phak-ch4-aerodynamics": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/phak/06_phak_ch4.pdf",
    "phak-ch7-instruments": "https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/phak/09_phak_ch7.pdf",
}

TEXT_EMBEDDER = "BAAI/bge-small-en-v1.5"          # text -> 384-dim
CLIP_EMBEDDER = "google/siglip2-base-patch16-224"  # image & text -> 768-dim shared space

DATA = Path("data")        # PDFs live here; the FAA chapters ship with the repo
CORPUS = Path("corpus")    # everything generated lands here (gitignored)
CHUNK_CHARS = 700          # ~150 tokens: small enough to be precise, big enough to mean something
MIN_IMAGE_SIDE = 220       # skip icons and page decorations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_pdfs() -> list[tuple[str, Path]]:
    """Every PDF already in data/, plus any PDF_URLS entry not yet downloaded."""
    DATA.mkdir(exist_ok=True)
    CORPUS.mkdir(exist_ok=True)
    for name, url in PDF_URLS.items():
        path = DATA / f"{name}.pdf"
        if not path.exists():
            print(f"downloading {name} ...")
            path.write_bytes(requests.get(url, timeout=120).content)
    return sorted((p.stem, p) for p in DATA.glob("*.pdf"))


def extract(doc_name: str, pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """Walk the PDF page by page; collect text chunks and real figures."""
    texts, images = [], []
    img_dir = CORPUS / "images"
    img_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    for page in doc:
        # --- text: paragraph blocks, greedily packed into ~CHUNK_CHARS chunks
        buf = ""
        for block in page.get_text("blocks"):
            paragraph = block[4].strip()
            if len(paragraph) < 40:                      # headers, page numbers
                continue
            if len(buf) + len(paragraph) > CHUNK_CHARS and buf:
                texts.append({"doc": doc_name, "page": page.number + 1, "text": buf})
                buf = ""
            buf = (buf + "\n" + paragraph).strip()
        if buf:
            texts.append({"doc": doc_name, "page": page.number + 1, "text": buf})

        # --- images: every embedded figure big enough to be a real diagram
        for xref, *_ in page.get_images(full=True):
            raw = doc.extract_image(xref)
            img = Image.open(io.BytesIO(raw["image"])).convert("RGB")
            if min(img.size) < MIN_IMAGE_SIDE:
                continue
            fname = f"{doc_name}-p{page.number + 1}-x{xref}.png"
            img.save(img_dir / fname)
            images.append({"doc": doc_name, "page": page.number + 1, "file": fname})
    return texts, images


@torch.inference_mode()
def embed_texts(texts: list[dict]) -> torch.Tensor:
    model = SentenceTransformer(TEXT_EMBEDDER, device=DEVICE)
    return model.encode(
        [t["text"] for t in texts],
        batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )


@torch.inference_mode()
def embed_images(images: list[dict]) -> torch.Tensor:
    model = AutoModel.from_pretrained(CLIP_EMBEDDER, torch_dtype=torch.float16).to(DEVICE)
    processor = AutoProcessor.from_pretrained(CLIP_EMBEDDER)
    feats = []
    for i in range(0, len(images), 32):
        batch = [Image.open(CORPUS / "images" / m["file"]) for m in images[i : i + 32]]
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
        f = model.get_image_features(**inputs)
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
    return torch.cat(feats).numpy()


def main() -> None:
    all_texts, all_images = [], []
    for name, path in collect_pdfs():
        texts, images = extract(name, path)
        print(f"{name}: {len(texts)} text chunks, {len(images)} figures")
        all_texts += texts
        all_images += images

    text_vecs = embed_texts(all_texts)
    image_vecs = embed_images(all_images)

    for vecs, fname in [(text_vecs, "text.faiss"), (image_vecs, "image.faiss")]:
        index = faiss.IndexFlatIP(vecs.shape[1])   # exact inner product = cosine (vectors are unit length)
        index.add(vecs)
        faiss.write_index(index, str(CORPUS / fname))

    (CORPUS / "chunks.json").write_text(json.dumps({"texts": all_texts, "images": all_images}))
    print(f"indexed {len(all_texts)} chunks + {len(all_images)} figures -> {CORPUS}/")


if __name__ == "__main__":
    main()
