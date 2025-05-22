from trl import SFTTrainer, SFTConfig

# define hyperparams
sft_config = SFTConfig(
        output_dir="graph-qa-sft",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=50,
        save_strategy="epoch"
        )

# define custom collator
def collate_fn(examples):
    # stack graph fts.
    node_feats = torch.stack([ex["node_feat"] for ex in examples])
    attn_masks = torch.stack([ex["attn_mask"] for ex in examples])

    # tokenize questions + answers
    questions = [ex["question"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    tok = tokenizer(
            questions,
            answers,
            padding=True,
            truncation=True,
            return_tensors="pt",
            )

    # mask padding tokens
    labels = tok["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {"node_feat": node_feats,
            "attn_mask": attn_masks,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"]
            "labels": labels,

# instantiate and run
trainer = SFTTrainer(
    model=graph_qa_model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    dataset_text_field="",
    )

trainer.train()
