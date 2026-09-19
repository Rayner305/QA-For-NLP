from config import *

#Reranker Class
class SciBERTReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(RERANKER_MODEL_NAME)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        h = out.last_hidden_state[:, 0, :]
        return self.fc(h).squeeze(-1)


#Metrics And Normalization
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0: return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def is_paragraph_match(retrieved, golds, threshold=0.5):
    p_toks = set(normalize_answer(retrieved).split())
    if not p_toks: return False
    for g in golds:
        g_toks = set(normalize_answer(g).split())
        if not g_toks: continue
        if len(p_toks.intersection(g_toks)) / len(g_toks) >= threshold: return True
    return False


#Main Evaluation
def evaluate():
    print("Loading Models from Config paths...")

    # Load Reranker
    try:
        reranker_tok = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
        reranker = SciBERTReranker().to(DEVICE)
        reranker.load_state_dict(torch.load(f"{RERANKER_MODEL_PATH}/pytorch_model.bin", map_location=DEVICE))
        reranker.eval()
        print("Reranker Model Loaded.")
    except Exception as e:
        print(f"Failed to load Reranker. Error: {e}")
        return

    #Load QA
    try:
        qa_tok = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH).to(DEVICE)
        qa_model.eval()
        print("QA Model Loaded.")
    except Exception as e:
        print(f"Failed to load QA Model. Error: {e}")
        return

    print("Starting Evaluation...")
    ds = load_dataset("allenai/qasper", split="validation")

    metrics = {
        "rr_1": 0, "rr_3": 0, "rr_50": 0, "rr_100": 0,
        "qa_f1_top1": [], "qa_em_top1": [],
        "qa_f1_top5": [], "qa_em_top5": [],
        "total": 0
    }

    TOP_K_QA = 5
    LIMIT_SAMPLES = None

    for idx, item in enumerate(ds):
        if LIMIT_SAMPLES and idx >= LIMIT_SAMPLES: break

        ft = item.get("full_text", {})
        if not isinstance(ft, dict): continue
        paras = [clean_text(p) for plist in ft.get("paragraphs", []) for p in plist if len(p.split()) > 15]

        qas = item["qas"]
        for i, q in enumerate(qas["question"]):
            q = clean_text(q)
            golds = [a["highlighted_evidence"][0] for a in qas["answers"][i].get("answer", []) if
                     "highlighted_evidence" in a and a["highlighted_evidence"]]
            if not golds: continue

            metrics["total"] += 1

            #Reranking
            scores = []
            with torch.no_grad():
                for p in paras:
                    inp = reranker_tok(q, p, return_tensors="pt", truncation=True, max_length=256,
                                       padding="max_length").to(DEVICE)
                    scores.append(reranker(inp["input_ids"], inp["attention_mask"]).item())
            order = np.argsort(scores)[::-1]

            #Reranking Metrics
            for rank, para_idx in enumerate(order[:100]):
                if is_paragraph_match(paras[para_idx], golds):
                    if rank == 0: metrics["rr_1"] += 1
                    if rank < 3: metrics["rr_3"] += 1
                    if rank < 50: metrics["rr_50"] += 1
                    if rank < 100: metrics["rr_100"] += 1
                    break

            #QA Extraction
            def get_ans(p):
                inp = qa_tok(q, p, return_tensors="pt", truncation=True, max_length=384, padding="max_length").to(DEVICE)
                with torch.no_grad():
                    out = qa_model(**inp)
                s_log, e_log = out.start_logits[0].cpu().numpy(), out.end_logits[0].cpu().numpy()
                best_s, best_e, best_sc = 0, 0, -1e9
                for s in range(len(s_log)):
                    for e in range(s, min(s + 40, len(e_log))):
                        if s_log[s] + e_log[e] > best_sc: best_sc, best_s, best_e = s_log[s] + e_log[e], s, e
                return qa_tok.decode(inp["input_ids"][0][best_s:best_e + 1], skip_special_tokens=True)

            pred_top1 = get_ans(paras[order[0]])
            metrics["qa_f1_top1"].append(max([compute_f1(g, pred_top1) for g in golds]))
            metrics["qa_em_top1"].append(max([compute_exact(g, pred_top1) for g in golds]))

            preds_top5 = [get_ans(paras[pidx]) for pidx in order[:TOP_K_QA]]
            metrics["qa_f1_top5"].append(max([compute_f1(g, p) for g in golds for p in preds_top5]))
            metrics["qa_em_top5"].append(max([compute_exact(g, p) for g in golds for p in preds_top5]))

        print(f"Processed {metrics['total']} questions...", end='\r')

    t = metrics["total"]
    if t == 0: return

    #Table Output
    print("\n" + "=" * 55)
    print(f"{'FINAL RESULTS':^55}")
    print(f"{f'Total Samples: {t}':^55}")
    print("=" * 55)
    print(f"{'METRIC':<30} | {'SCORE':>20}")
    print("-" * 55)
    # Reranker
    print(f"{'Reranker Recall@1':<30} | {metrics['rr_1'] / t * 100:>19.2f}%")
    print(f"{'Reranker Recall@3':<30} | {metrics['rr_3'] / t * 100:>19.2f}%")
    print(f"{'Reranker Recall@50':<30} | {metrics['rr_50'] / t * 100:>19.2f}%")
    print(f"{'Reranker Recall@100':<30} | {metrics['rr_100'] / t * 100:>19.2f}%")
    print("-" * 55)
    # QA
    print(f"{'QA Top-1 Exact Match':<30} | {np.mean(metrics['qa_em_top1']) * 100:>19.2f}%")
    print(f"{'QA Top-1 F1 Score':<30} | {np.mean(metrics['qa_f1_top1']) * 100:>19.2f}%")
    print(f"{'QA Top-5 Exact Match':<30} | {np.mean(metrics['qa_em_top5']) * 100:>19.2f}%")
    print(f"{'QA Top-5 F1 Score':<30} | {np.mean(metrics['qa_f1_top5']) * 100:>19.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()
