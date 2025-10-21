import pdfToChunks as PDF
import VectorDB as VDB
import torch
import os
import warnings
import faiss
from pathlib import Path
from datasets import load_dataset
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed

def dataset_from_jsonl(p):
    ds_dict = load_dataset("json", data_files=p)
    return ds_dict["train"]

def to_io(msgs):
    sys = usr = ans = ""
    for m in msgs:
        r, t = m["role"], (m["content"] or "").strip()
        if r == "system":
            sys = t
        elif r == "user":
            usr = t
        elif r == "assistant":
            ans = t
    src = "\n".join(filter(None, [f"System: {sys}" if sys else "", f"User: {usr}" if usr else ""]))
    return {"source": src, "target": ans}

def preprocess(b, tok, MAX_SOURCE_LEN, MAX_TARGET_LEN):
    x = tok(b["source"], max_length=MAX_SOURCE_LEN, truncation=True, padding=False)
    y = tok(text_target=b["target"], max_length=MAX_TARGET_LEN, truncation=True, padding=False)
    x["labels"] = y["input_ids"]
    return x


def train_model(DATA_DIR=Path("/content/data"), pdf_path="/operating_systems_three_easy_pieces.pdf",
                tok_checkpoint="sentence-transformers/all-MiniLM-L6-v2", MODEL_NAME="google/flan-t5-xl",
                TRAIN_JSONL="/studybuddy_sft_longform_train.jsonl", VAL_JSONL="/studybuddy_sft_longform_val.jsonl",
                OUTPUT_DIR="/content/studybuddy-t5-fullft", MAX_SOURCE_LEN=1536, MAX_TARGET_LEN=512):

    DATA_DIR.mkdir(exist_ok=True)
    chunks_path = DATA_DIR / "chunks.jsonl"
    index_path = DATA_DIR / "index.faiss"
    meta_path = DATA_DIR / "meta.pkl"

    st_tok = AutoTokenizer.from_pretrained(tok_checkpoint, use_fast=True)
    PDF.write_chunks(pdf_path, st_tok, str(chunks_path))
    texts, metas = VDB.read_chunks(chunks_path)
    stModel = VDB.build_st_model()
    embeddings = VDB.create_normalized_embeddings(texts, stModel)
    dim = embeddings.shape[1]
    index = VDB.create_FIASS(dim)
    index = VDB.FAISS_prep_and_add(embeddings, index)
    VDB.save_FIASS(index)
    faiss.write_index(index, str(index_path))
    VDB.save_meta(metas, texts, st_tok, out_path=meta_path)

    warnings.filterwarnings("ignore")

    assert torch.cuda.is_available(), "❌ Please enable GPU: Runtime → Change runtime type → GPU"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    set_seed(42)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_ds = dataset_from_jsonl(TRAIN_JSONL)
    val_ds = dataset_from_jsonl(VAL_JSONL)

    train_ds = train_ds.map(lambda ex: to_io(ex["messages"])).filter(lambda ex: ex["target"].strip() != "")
    val_ds = val_ds.map(lambda ex: to_io(ex["messages"])).filter(lambda ex: ex["target"].strip() != "")

    gen_tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False

    train_tok = train_ds.map(
        lambda b: preprocess(b, gen_tok, MAX_SOURCE_LEN, MAX_TARGET_LEN),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    val_tok = val_ds.map(
        lambda b: preprocess(b, gen_tok, MAX_SOURCE_LEN, MAX_TARGET_LEN),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    collator = DataCollatorForSeq2Seq(gen_tok, model=model, label_pad_token_id=-100)

    small_val = val_tok.select(range(min(1000, len(val_tok))))  # change 1000 to taste

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1.5e-5,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=24,
        gradient_checkpointing=True,
        bf16=True,
        optim="adafactor",

        eval_strategy="steps",
        eval_steps=2000,
        predict_with_generate=False,

        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,

        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=100,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        remove_unused_columns=False,
        include_num_input_tokens_seen=False,

        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=small_val,
        tokenizer=gen_tok,
        data_collator=collator
    )

    trainer.train()
    return model, trainer, val_tok, gen_tok

def save_model(model, tok, OUTPUT_DIR="/content/studybuddy-t5-fullft"):

    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

def model_validation(model, trainer, val_tok):
    """
    Evaluate model preformance on validation set.
    :param model: LLM model.
    :param trainer: Model Trainer.
    :param val_tok: Validation Tokenizer.
    """

    model.eval()

    metrics = trainer.evaluate(eval_dataset=val_tok, metric_key_prefix="eval")
    print("Final evaluation (loss-only):", metrics)

    if "eval_loss" in metrics and metrics["eval_loss"] < float("inf"):
        print("Perplexity:", math.exp(metrics["eval_loss"]))
