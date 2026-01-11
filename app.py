from config import *

#just fro view :)
nltk.download('punkt', quiet=True)
stemmer = PorterStemmer()


def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class SciBERTReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        h = out.last_hidden_state[:, 0, :]
        return self.fc(h).squeeze(-1)


print("Loading Models...Please Wait")
QA_MODEL_PATH = f"{OUTPUT_DIR}/qa_model"
RERANKER_MODEL_PATH = f"{OUTPUT_DIR}/reranker_model"

try:
    tokenizer_qa = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
    model_qa = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH).to(DEVICE)
    model_qa.eval()
    print("QA Model-(DeBERTa) Loaded.")
except Exception as e:
    print(f"Some Thing Rong ,We Can't Download the QA Model-(DeBERTa) : {e}")
    exit()

try:
    tokenizer_rerank = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
    model_rerank = SciBERTReranker().to(DEVICE)
    model_rerank.load_state_dict(torch.load(f"{RERANKER_MODEL_PATH}/pytorch_model.bin", map_location=DEVICE))
    model_rerank.eval()
    print("Reranker Model-(SciBERT) Loaded Successfully.")
except Exception as e:
    print(f"Error loading Reranker: {e}")
    exit()

print("Building Search Index-(BM25)...")
raw_dataset = load_dataset("allenai/qasper", split="validation")
corpus_global = []
for item in raw_dataset:
    ft = item.get("full_text", {})
    if isinstance(ft, dict):
        for plist in ft.get("paragraphs", []):
            for p in plist:
                p_clean = clean_text(p)
                if len(p_clean) > 50:
                    corpus_global.append(p_clean)

if len(corpus_global) > 10000:
    print(f"Dataset is large, indexing first 10,000 paragraphs for speed.")
    corpus_global = corpus_global[:10000]

tokenized_corpus_global = [[stemmer.stem(w) for w in doc.split()] for doc in corpus_global]
bm25_global = BM25Okapi(tokenized_corpus_global)
print(f"Index Ready:({len(corpus_global)} paragraphs).")


def get_user_answer(question):
    if not question: return "Please ask a question.", ""
    q_clean = clean_text(question)
    q_tokens = [stemmer.stem(w) for w in q_clean.split()]

    retrieved = bm25_global.get_top_n(q_tokens, corpus_global, n=30)

    rerank_scores = []
    with torch.no_grad():
        for doc in retrieved:
            inputs = tokenizer_rerank(q_clean, doc, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            score = model_rerank(inputs['input_ids'], inputs['attention_mask']).item()
            rerank_scores.append((score, doc))
    rerank_scores.sort(key=lambda x: x[0], reverse=True)
    best_context = rerank_scores[0][1]

    inputs = tokenizer_qa(q_clean, best_context, return_tensors="pt", truncation=True, max_length=384).to(DEVICE)
    with torch.no_grad():
        out = model_qa(**inputs)

    s_logits = out.start_logits[0].cpu().numpy()
    e_logits = out.end_logits[0].cpu().numpy()
    best_score = -float('inf')
    best_s, best_e = 0, 0
    for s in range(len(s_logits)):
        for e in range(s, min(len(s_logits), s + 40)):
            score = s_logits[s] + e_logits[e]
            if score > best_score:
                best_score = score
                best_s = s
                best_e = e
    pred_ans = tokenizer_qa.decode(inputs.input_ids[0][best_s:best_e + 1], skip_special_tokens=True)

    if not pred_ans.strip() or "[CLS]" in pred_ans or "[SEP]" in pred_ans or best_score < 1.0:
        return "No direct answer found within high confidence :(.", best_context
    return pred_ans, best_context


custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #00c6ff, #0072ff, #f72585, #b5179e);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    font-family: 'Roboto', sans-serif;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
#header {
    color: white;
    text-align: center;
    margin-top: 3rem;
    margin-bottom: 2rem;
}
#header h1 {
    font-size: 4rem !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
    margin-bottom: 0.5rem;
}
#header h3 {
    font-size: 1.8rem !important;
    font-weight: 300;
    opacity: 0.9;
}
.search-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    margin-bottom: 3rem;
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}
.faq-header {
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    font-size: 2rem;
    font-weight: bold;
}
.faq-container {
    max-width: 900px;
    margin: auto;
    margin-bottom: 3rem;
}
.accordion-header {
    background: white !important;
    color: #333 !important;
    border-radius: 12px !important;
    margin-bottom: 0.8rem !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    font-weight: 500;
}
.footer {
    text-align: center;
    color: white;
    margin-top: 4rem;
    margin-bottom: 2rem;
}
.email-btn {
    background-color: #fff !important;
    color: #0072ff !important;
    border: none !important;
    font-weight: bold;
    padding: 0.8rem 2rem;
    border-radius: 30px;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.email-btn:hover {
    transform: scale(1.05);
}
.search-btn {
    background-color: #0072ff !important;
    color: white !important;
    border: none !important;
    font-weight: bold;
    border-radius: 8px; 
    height: 100%;
}
.search-btn:hover {
    background-color: #005bb5 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# QA For NLP")
        gr.Markdown("### How can we help you?")

    with gr.Column(elem_classes="search-container"):
        with gr.Row():
            q_input = gr.Textbox(
                label="",
                placeholder="Ask a question about scientific papers...",
                show_label=False,
                container=False,
                scale=4
            )
            search_btn = gr.Button("Search", variant="primary", scale=1, elem_classes="search-btn")

        with gr.Column(visible=False) as results_col:
            gr.Markdown("### Our Answer:")
            ans_output = gr.Textbox(show_label=False, lines=3, show_copy_button=True)
            gr.Markdown("### Source Context:")
            ctx_output = gr.Textbox(show_label=False, lines=6, show_copy_button=True)

        def search_and_display(q):
            ans, ctx = get_user_answer(q)
            return {results_col: gr.update(visible=True),
                    ans_output: gr.update(value=ans),
                    ctx_output: gr.update(value=ctx)}

        search_btn.click(
            search_and_display,
            inputs=q_input,
            outputs=[results_col, ans_output, ctx_output]
        )
        q_input.submit(
            search_and_display,
            inputs=q_input,
            outputs=[results_col, ans_output, ctx_output]
        )

    gr.Markdown("Frequently Asked Questions", elem_classes="faq-header")
    with gr.Column(elem_classes="faq-container"):
        with gr.Accordion("What is the main advantage of this hybrid system?", open=False,
                          elem_classes="accordion-header"):
            gr.Markdown(
                "This system combines SciBERT's specialized vocabulary for effective retrieval of scientific texts with DeBERTa's superior language understanding for accurate answer extraction.")

        with gr.Accordion("What dataset was this model trained on?", open=False, elem_classes="accordion-header"):
            gr.Markdown(
                "The models were trained and evaluated on the Qasper dataset, which contains over **5,049 questions** extracted from **1,585 academic NLP papers**.")

        with gr.Accordion("How accurate is the system?", open=False, elem_classes="accordion-header"):
            gr.Markdown("""
                            On the Qasper validation set (**927 samples**), the system achieved the following academic results:

                            | Component | Metric | Score |
                            | :--- | :--- | :--- |
                            | **Retrieval (Search)** | Recall@1 | 46.82% |
                            | | Recall@3 | 73.89% |
                            | | Recall@50 | **95.79%** |
                            | | Recall@100 | 95.79% |
                            | **QA (Reader)** | Top-1 F1 Score | 29.91% |
                            | | Top-5 F1 Score | **48.89%** |
                            | | Top-1 Exact Match | 10.57% |
                            | | Top-5 Exact Match | 22.33% |

                            you know Recall@50 mean the correct document is retrieved 95.79% of the time.*
                            """)

    with gr.Column(elem_classes="footer"):
        gr.Markdown("### Can't find what you are looking for?")
        email_btn = gr.Button("Email Us", elem_classes="email-btn")
        email_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { window.location.href = 'mailto:202310819@ammanu.edu.jo?subject=Question about QA System'; }"
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True, share=True)