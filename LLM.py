from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def create_CE_model(model_checkpoint="cross-encoder/ms-marco-electra-base", maxLength=512):
    """
    Create Cross encoder model
    :param model_checkpoint: Model Checkpoint.
    :param maxLength: Max length.
    :return: Cross encoder model.
    """

    model = CrossEncoder(model_checkpoint, max_length=maxLength)

    return model


def rerank(model, hits, question, top_k=5, batch_size=32):
    """
    Reranks FIASS index similarity search result using Cross encoder model.
    :param model: Cross encoder model.
    :param hits: FIASS index similarity search index results.
    :param question: Question.
    :param top_k: Number of top k results to select.
    :param batch_size: Number of predictions to process at once.
    :return: Top k similar texts.
    """

    pairs = [(question, h["text"]) for h in hits]
    scores = model.predict(pairs, batch_size=batch_size)

    for i in range(len(hits)):
        hits[i]["ce_score"] = float(scores[i])

    hits = sorted(hits, key=lambda x: x["ce_score"], reverse=True)

    return hits[:top_k]


def build_LLM_and_tokenizer(model_checkpoint="google/flan-t5-xl"):
    """
    Build Flan-t5 LLM and tokenizer.
    :param model_checkpoint: Model checkpoint.
    :return: Tokenizer and model.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    return tokenizer, model


def generate_output(model, tokenizer, question, hits, top_m=5, max_context=4000):
    """
    Use LLM model in conjuction with results from FIASS Query to answer question.
    :param model: LLM model.
    :param tokenizer: LLM Tokenizer.
    :param question: Users Quetion
    :param hits: Results from FIASS index search using Question as input.
    :param top_m: The number of the top m hits to build answer from.
    :param max_context: Max length of output.
    :return: Answer to question + citations.
    """

    context = ""
    citations = ""

    for h in hits[:top_m]:
        context += h["text"]
        context += ". "

        citations += (h["doc_name"] + " " + str(h["slide_id"])) + ", "

    prompt = (
        "Answer the question using the context. "
        "Write 2–4 concise sentences. Cite as (Slide N). "
        "If missing, reply: Not found in slides.\n"
        f"Question: {question}\nContext:\n{context}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=min(max_context, tokenizer.model_max_length))
    model.eval()

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, num_beams=6,
                             length_penalty=1, no_repeat_ngram_size=3)
    out = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    return out + "\nCitations: " + citations.rstrip(", ")
