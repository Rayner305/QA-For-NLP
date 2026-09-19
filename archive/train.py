from config import *

# The Reranker Class
class SciBERTReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(RERANKER_MODEL_NAME)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.encoder(ids, mask)
        h = out.last_hidden_state[:, 0, :]
        return self.fc(h).squeeze(-1)


#Training the Reranker
def train_reranker(raw_ds):
    print("\nStep 1: Training Reranker... This might take a while, go take a cup coffee.")
    pairs = []

    #loop through the data
    for item in raw_ds:
        ft = item.get("full_text", {})
        if not isinstance(ft, dict): continue
        paras = [clean_text(p) for plist in ft.get("paragraphs", []) for p in plist if len(p.split()) > 15]
        if len(paras) < 3: continue

        qas = item["qas"]
        for i in range(len(qas["question"])):
            q = clean_text(qas["question"][i])
            for a in qas["answers"][i].get("answer", []):
                if "highlighted_evidence" in a and a["highlighted_evidence"]:
                    span = clean_text(a["highlighted_evidence"][0])
                    pos = next((p for p in paras if span in p), None)
                    if not pos: continue

                    #We need bad examples to teach the model what is wrong
                    neg_candidates = [p for p in paras if p != pos]
                    if len(neg_candidates) >= 3:
                        for neg in random.sample(neg_candidates, 3):
                            pairs.append((q, pos, neg))

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    model = SciBERTReranker().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_RERANKER)

    class PairDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    loader = DataLoader(PairDataset(pairs), batch_size=BATCH_SIZE, shuffle=True)
    total_steps = (len(pairs) // BATCH_SIZE) * RERANKER_EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, 100, total_steps)

    model.train()
    for epoch in range(RERANKER_EPOCHS):
        for step, (q, p, n) in enumerate(loader):
            pos = tokenizer(q, p, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            neg = tokenizer(q, n, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)


            pos_score = model(pos['input_ids'], pos['attention_mask'])
            neg_score = model(neg['input_ids'], neg['attention_mask'])
            diff = pos_score - neg_score
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                opt.step();
                scheduler.step();
                opt.zero_grad()

            if step % 200 == 0: print(f"   Epoch {epoch + 1} | Loss: {loss.item() * GRAD_ACCUM:.4f} (Still working...)", end='\r')

    save_path = os.path.join(OUTPUT_DIR, "reranker_model")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_path)
    print("\nReranker Model saved safely! :)")


#Training the QA Model-(The Smart Part)
def train_qa(raw_ds):
    print("\nStep 2: Training QA Model...")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME, use_fast=True)

    samples = []
    for item in raw_ds:
        ft = item.get("full_text", {})
        if not isinstance(ft, dict): continue
        paras = [clean_text(p) for plist in ft.get("paragraphs", []) for p in plist if len(p.split()) > 15]

        qas = item["qas"]
        for i in range(len(qas["question"])):
            q = clean_text(qas["question"][i])
            for a in qas["answers"][i].get("answer", []):
                if "highlighted_evidence" in a and a["highlighted_evidence"]:
                    span = clean_text(a["highlighted_evidence"][0])
                    for para in paras:
                        start = para.find(span)
                        if start != -1:
                            samples.append(
                                {"question": q, "context": para, "answers": {"text": [span], "answer_start": [start]}})
                            break

    qa_ds = HFDataset.from_list(samples)

    # This function is fixed to stop the annoying Index Error
    def tokenize(ex):
        toks = tokenizer(ex["question"], ex["context"], truncation="only_second", max_length=MAX_LEN, stride=STRIDE,
                         return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
        sample_map = toks.pop("overflow_to_sample_mapping")
        offset_map = toks.pop("offset_mapping")
        toks["start_positions"], toks["end_positions"] = [], []

        for i, offsets in enumerate(offset_map):
            input_ids = toks["input_ids"][i]
            cls_index = 0
            seq_ids = toks.sequence_ids(i)
            ans = ex["answers"][sample_map[i]]

            if not ans["answer_start"] or ans["answer_start"][0] == -1:
                toks["start_positions"].append(cls_index);
                toks["end_positions"].append(cls_index)
                continue

            s_char = ans["answer_start"][0]
            e_char = s_char + len(ans["text"][0])

            # Checking boundaries so the code does not crash
            idx_start = 0
            while idx_start < len(input_ids) and seq_ids[idx_start] != 1: idx_start += 1

            idx_end = len(input_ids) - 1
            while idx_end >= 0 and seq_ids[idx_end] != 1: idx_end -= 1

            if idx_start >= len(input_ids) or idx_end < 0:
                toks["start_positions"].append(cls_index);
                toks["end_positions"].append(cls_index)
                continue

            if not (offsets[idx_start][0] <= s_char and offsets[idx_end][1] >= e_char):
                toks["start_positions"].append(cls_index);
                toks["end_positions"].append(cls_index)
            else:
                while idx_start < len(offsets) and offsets[idx_start][0] <= s_char: idx_start += 1
                toks["start_positions"].append(idx_start - 1)


                while idx_end >= 0 and offsets[idx_end][1] >= e_char: idx_end -= 1
                toks["end_positions"].append(idx_end + 1)
        return toks

    encoded = qa_ds.map(tokenize, batched=True, remove_columns=qa_ds.column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME).to(DEVICE)
    args = TrainingArguments(os.path.join(OUTPUT_DIR, "checkpoints_qa"), learning_rate=LR_QA,
                             per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                             num_train_epochs=QA_EPOCHS, fp16=torch.cuda.is_available(), save_strategy="epoch",
                             report_to="none")

    Trainer(model=model, args=args, train_dataset=encoded, tokenizer=tokenizer).train()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "qa_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "qa_model"))
    print("\nQA Model Saved. We are done here.")


if __name__ == "__main__":
    print(f"Starting the Training Process... It's Take Long Time")
    ds = load_dataset("allenai/qasper", split="train")

    train_reranker(ds)
    print("Reranker Training is done.")

    train_qa(ds)
    print("\nFinally! Everything is finished.")
