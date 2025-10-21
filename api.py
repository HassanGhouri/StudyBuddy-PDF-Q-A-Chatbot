import pickle
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import LLM as LLM
import pdfToChunks as PDF
import VectorDB as VDB
from transformers import AutoTokenizer


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
chunks_path = DATA_DIR / "chunks.jsonl"
index_path = DATA_DIR / "index.faiss"
meta_path = DATA_DIR / "meta.pkl"

emb_model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
ce_model_chekpoint = "cross-encoder/ms-marco-electra-base"
gen_model_checkpoint = "google/flan-t5-xl"

api = FastAPI(title="StudyBuddy API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index_obj = None
metas: List[Dict] = []
texts: List[str] = []
st_model = None
ce_model = None
tokenizer = None
llm = None

def _load_or_build_runtime():
    """Load or reload model with updated FAISS and metas."""

    global index_obj, metas, texts, st_model, ce_model, tokenizer, llm

    if meta_path.exists():

        with meta_path.open("rb") as f:
            payload = pickle.load(f)

        metas, texts = payload["metas"], payload["texts"]
        encoder_name = payload.get("encoder", emb_model_checkpoint)

    else:
        metas, texts = [], []
        encoder_name = emb_model_checkpoint

    if index_path.exists():
        index_obj = faiss.read_index(str(index_path))

    else:
        index_obj = None

    st_model = VDB.build_st_model(encoder_name)
    ce_model = LLM.create_CE_model(ce_model_chekpoint)
    tokenizer, llm=LLM.build_LLM_and_tokenizer(gen_model_checkpoint)

class AskIn(BaseModel):
    question: str
    k: int = 8

@api.on_event("startup")
def _startup():
    _load_or_build_runtime()


@api.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF, write chunks, embedd then, store in FAISS, save metas."""

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF")

    up_path = DATA_DIR / file.filename
    up_path.write_bytes(await file.read())

    tok = AutoTokenizer.from_pretrained(emb_model_checkpoint, use_fast=True)
    PDF.write_chunks(str(up_path), tok, str(chunks_path))

    texts, metas = VDB.read_chunks(chunks_path)

    if not texts:
        raise HTTPException(status_code=500, detail="No text extracted.")

    st = VDB.build_st_model(emb_model_checkpoint)
    emb = VDB.create_normalized_embeddings(texts, st)

    if emb is None or getattr(emb, "shape", (0, 0))[0] == 0:
        raise HTTPException(status_code=500, detail="No embeddings produced.")

    index = VDB.create_FIASS(emb.shape[1])
    VDB.FAISS_prep_and_add(emb, index)
    faiss.write_index(index, str(index_path))

    with meta_path.open("wb") as f:
        pickle.dump({"metas": metas, "texts": texts, "encoder": emb_model_checkpoint, "normalized": True}, f)

    _load_or_build_runtime()

    return {"status": "ok", "doc_name": up_path.stem, "chunks": len(texts)}

@api.post("/ask")
def ask(inp: AskIn):
    """Ask a question, retrieve relevent chunks, rerank them based on relevence, generate answer + citations."""

    if index_obj is None or not texts:
        raise HTTPException(status_code=400, detail="No index loaded. Ingest a PDF first.")

    initial = VDB.search(inp.question, st_model, index_obj, metas, texts, k=inp.k)

    if not initial:
        return {"answer": "Not found in slides.", "citations": [], "hits": []}

    hits = LLM.rerank(ce_model, initial, inp.question, top_k=5)
    answer = LLM.generate_output(llm, tokenizer, inp.question, hits)
    citations = [f"({h.get('doc_name','Document')} • Slide {h.get('slide_id','?')})" for h in hits]
    compact_hits = []

    for h in hits[:5]:
        compact_hits.append({
            "doc_name": h.get("doc_name"),
            "slide_id": h.get("slide_id"),
            "chunk_id": h.get("chunk_id"),
            "text": (h.get("text") or "")[:1200]
        })

    return {"answer": answer, "citations": citations, "hits": compact_hits}
