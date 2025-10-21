import faiss
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np


def build_st_model(model_checkpoint="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Build Sentence Transformer model.
    :param model_checkpoint: Model checkpoint.
    :return: ST Model
    """

    st_model = SentenceTransformer(model_checkpoint)

    return st_model


def create_normalized_embeddings(input, st_model, batch_size=64):
    """
    Create Normalized embeddings.
    :param input: Input text.
    :param st_model: Sentence Transformer model
    :param batch_size: Batch size.
    :return: Normalized embeddings of text.
    """

    embeddings = st_model.encode(input, batch_size=batch_size, normalize_embeddings=True)

    return embeddings


def create_FIASS(dim):
    """
    Create FIASS index for efficient similarity search.
    :param dim: Dimension of index.
    :return: FIASS index.
    """

    index = faiss.IndexFlatIP(dim)

    return index


def FAISS_prep_and_add(embeddings, index):
    """
    Prep embeddings and add to index.
    :param embeddings: Normalized text embeddings.
    :param index: FIASS index.
    :return: Return updated index.
    """

    embeddings = embeddings.astype("float32")
    index.add(embeddings)

    return index


def save_FIASS(index, index_out_path="/index.faiss"):
    """
    Save FIASS index.
    :param index: FIASS index.
    :param index_out_path: Path to file to write to, default=/index.faiss.
    :return: NONE.
    """

    faiss.write_index(index, index_out_path)


def load_FIASS(index_out_path="/index.faiss"):
    """
    Load FIASS index from file.
    :param index_out_path: Path to FIASS index file, default=/index.fiass.
    :return: Index
    """

    index = faiss.read_index(index_out_path)

    return index


def read_chunks(path: Path):
    """
    Read chunks.jsonl file and return text and metas data.
    :param path: Path to chunks.jsonl file.
    :return: Text and Metas data.
    """

    texts, metas = [], []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metas.append({
                "doc_name": obj["doc_name"],
                "slide_id": obj["slide_id"],
                "chunk_id": obj["chunk_id"],
            })

    return texts, metas


def save_meta(metas, texts, encoder_name, out_path="/meta.pkl"):
    """
    Save meta data to file.
    :param metas: Meta data.
    :param texts: Text data.
    :param encoder_name: Encoder name.
    :param out_path: File meta daa is being written to, default=data/meta.pkl.
    :return: NONE
    """

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(
            {"metas": metas, "texts": texts, "encoder": encoder_name, "normalized": True},
            f,
        )


def load_meta(path="/meta.pkl"):
    """
    Load Meta data from file.
    :param path: Path to file containing meta data, default=data/meta/pkl
    :return: Read meta data.
    """

    with open(path, "rb") as f:

        return pickle.load(f)


def search(question: str, st_model, index, metas: list[dict], texts: list[str], k: int = 10):
    """
    Similarity search of FIASS vectorBD. Converts Question into vector then preforms similarity search using inner
    product to get cosine similarity, possible because vectors are normalized, returns 5 most similar vectors.
    vectors most similar
    :param question: Question.
    :param st_model: Sentence Transformer model.
    :param index: FIASS index.
    :param metas: Meta data.
    :param texts: Text data.
    :param k: number of results to output.
    :return: Top k similar chunks.
    """

    if index.ntotal == 0:

        return []

    q = st_model.encode([question], normalize_embeddings=True).astype(np.float32)
    kk = min(max(k * 3, k), index.ntotal)
    score, row_indices = index.search(q, kk)
    hits = []
    seen = set()

    for idx, score in zip(row_indices[0], score[0]):

        if idx < 0:
            continue

        m = metas[idx]
        key = (m.get("doc_id"), m.get("doc_name"), m["slide_id"], m["chunk_id"])

        if key in seen:
            continue

        seen.add(key)
        item = {
            "rank": len(hits),
            "score": float(score),
            "doc_name": m.get("doc_name"),
            "slide_id": m["slide_id"],
            "chunk_id": m["chunk_id"],
        }

        if texts is not None:
            t = texts[idx]
            item["text"] = t

        hits.append(item)

        if len(hits) >= k:
            break

    return hits


def format_citation(hit: dict) -> str:

    dn = hit.get("doc_name") or "Document"

    return f"({dn} • Slide {hit['slide_id']})"
