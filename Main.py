import trainingBuilding as TB
import os
import time
import re
import subprocess
import pathlib
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LOGS_DIR = "/content/_run_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def must_exist(path, hint=""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path} {hint}")

def start_bg(cmd, log_file):
    f = open(log_file, "w")
    print("▶️", " ".join(cmd), "(log:", log_file, ")")

    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True), f

def wait_for_tunnel(log_file, timeout=45):
    for _ in range(timeout):
        try:
            txt = open(log_file).read()
            m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", txt)
            if m:
                return m.group(0)
        except Exception:
            pass
        time.sleep(1)

    return None

def ensure_cloudflared():
    if shutil.which("cloudflared"):
        return "cloudflared"

    if pathlib.Path("./cloudflared").exists():
        return "./cloudflared"

    import urllib.request
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, "cloudflared")
    os.chmod("cloudflared", 0o755)

    return "./cloudflared"

def bootstrap_deps():

    try:
        import fastapi
        import uvicorn
        import streamlit
        import sentence_transformers
        import faiss

    except Exception:
        os.system("pip -q install fastapi uvicorn[standard] streamlit "
                  "transformers==4.44.2 accelerate sentence-transformers faiss-cpu "
                  "pypdf pdf2image pytesseract")

    if shutil.which("apt"):
        os.system("apt -yqq install tesseract-ocr poppler-utils > /dev/null || true")


def build():
    pdf_path = "/operating_systems_three_easy_pieces.pdf"
    model, trainer, val_tok, gen_tok = TB.train_model(pdf_path=pdf_path)
    TB.model_validation(model, trainer, val_tok)
    TB.save_model(model, gen_tok)

def save_model():
    pass

def reload_model(MODEL_PATH="/content/studybuddy-t5-fullft"):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    return model, tok

def run_model(MODEL_DIR="/content/studybuddy-t5-fullft", DATA_DIR="/content/data", API_PATH="/content/api.py",
              GUI_PATH="/content/GUI.py", API_HOST="0.0.0.0", API_PORT=8000, GUI_PORT=8501):

    bootstrap_deps()
    cfd_bin = ensure_cloudflared()

    os.system("pkill -f 'uvicorn|streamlit|cloudflared' || true")

    for f in ["config.json", "model.safetensors.index.json", "tokenizer.json"]:
        must_exist(os.path.join(MODEL_DIR, f), "(model folder not complete?)")

    must_exist(os.path.join(DATA_DIR, "index.faiss"), "(FAISS index not found)")
    must_exist(os.path.join(DATA_DIR, "meta.pkl"), "(meta.pkl not found)")
    must_exist(API_PATH, "(api.py not found)")
    must_exist(GUI_PATH, "(GUI.py not found)")

    with open(API_PATH, "r", encoding="utf-8") as f:
        api_txt = f.read()

    api_txt_patched = re.sub(
        r'gen_model_checkpoint\s*=\s*["\'].*?["\']',
        f'gen_model_checkpoint = "{MODEL_DIR}"',
        api_txt,
        count=1,
    )

    if api_txt_patched != api_txt:
        with open(API_PATH, "w", encoding="utf-8") as f:
            f.write(api_txt_patched)
        print("Patched api.py points to gen_model_checkpoint =", MODEL_DIR)
    else:
        print("api.py already points to:", MODEL_DIR)

    api_log = os.path.join(LOGS_DIR, "api.log")
    api_cmd = [
        "uvicorn", "api:api",
        "--app-dir", os.path.dirname(API_PATH) or ".",
        "--host", API_HOST, "--port", str(API_PORT),
        "--reload", "--workers", "1"
    ]
    api_proc, api_log_f = start_bg(api_cmd, api_log)
    time.sleep(3)

    api_tunnel_log = os.path.join(LOGS_DIR, "api_tunnel.log")
    api_tunnel_proc, api_tunnel_log_f = start_bg(
        [cfd_bin, "tunnel", "--url", f"http://localhost:{API_PORT}", "--no-autoupdate"],
        api_tunnel_log)
    api_url = wait_for_tunnel(api_tunnel_log)
    print("API public URL:", api_url or "(still starting…)")

    os.environ["API_URL"] = f"http://localhost:{API_PORT}"
    gui_log = os.path.join(LOGS_DIR, "gui.log")
    gui_cmd = ["streamlit", "run", GUI_PATH, "--server.port", str(GUI_PORT), "--server.headless", "true"]
    gui_proc, gui_log_f = start_bg(gui_cmd, gui_log)
    time.sleep(5)

    gui_tunnel_log = os.path.join(LOGS_DIR, "gui_tunnel.log")
    gui_tunnel_proc, gui_tunnel_log_f = start_bg(
        [cfd_bin, "tunnel", "--url", f"http://localhost:{GUI_PORT}", "--no-autoupdate"],
        gui_tunnel_log)
    gui_url = wait_for_tunnel(gui_tunnel_log)
    print("🖥️ GUI public URL:", gui_url or "(still starting…)")

    _open_logs = [api_log_f, api_tunnel_log_f, gui_log_f, gui_tunnel_log_f]
    _procs = [api_proc, api_tunnel_proc, gui_proc, gui_tunnel_proc]

    print(f"\n✅ Everything launched.\nOpen this GUI link → {gui_url}\n"
          "If 'Request failed', ensure API_URL points to the same Colab runtime.\n"
          "Stop later with: !pkill -f 'uvicorn|streamlit|cloudflared'")
