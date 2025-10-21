import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="StudyBuddy", layout="wide")
st.title("StudyBuddy — PDF Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_citations" not in st.session_state:
    st.session_state.last_citations = []

if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

with st.sidebar:
    st.header("Citations & Evidence")

    if st.session_state.last_hits:

        for i, h in enumerate(st.session_state.last_hits, start=1):
            label = f"{i}. ({h.get('doc_name','Document')} • Slide {h.get('slide_id','?')})"

            with st.expander(label, expanded=False):
                st.write(h.get("text", "").strip() or "_No text available_")
    else:
        st.info("Ask a question to see citations and snippets here.")

    st.divider()
    st.subheader("Upload a PDF")
    pdf = st.file_uploader("Choose a PDF", type=["pdf"])

    if st.button("Ingest PDF", use_container_width=True):

        if not pdf:
            st.warning("Please choose a PDF first.")

        else:
            try:
                files = {"file": (pdf.name, pdf.getvalue(), "application/pdf")}
                r = requests.post(f"{API}/ingest", files=files, timeout=300)

                if r.ok:
                    st.success(r.json().get("status", "done"))

                else:
                    st.error(f"Ingest failed: {r.status_code} {r.text}")

            except Exception as e:
                st.error(f"Ingest error: {e}")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.session_state.last_hits = []
        st.experimental_rerun()

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question about your PDF(s)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": [], "hits": []})

    with st.chat_message("user"):
        st.write(prompt)

    try:
        payload = {"question": prompt, "k": 8}
        r = requests.post(f"{API}/ask", json=payload, timeout=120)

        if not r.ok:
            raise RuntimeError(f"API error: {r.status_code} {r.text}")

        data = r.json()

        answer = data.get("answer", "")
        citations = data.get("citations", []) or []
        hits = data.get("hits", []) or []

        st.session_state.last_citations = citations
        st.session_state.last_hits = hits

        with st.chat_message("assistant"):
            st.write(answer)

            if citations:
                st.caption("Citations: " + "  ".join(citations))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations, "hits": hits}
        )

    except Exception as e:
        err = f"Request failed: {e}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err, "citations": [], "hits": []})
