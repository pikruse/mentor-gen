import os
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from utils.model import GraphQATransformer
from utils.dataloader import GraphQADataset

# paths
TEST_TSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'n50_edge_tests.tsv')

# initialize tokenizer and model
tokenizer = GraphQATransformer().tokenizer
model = GraphQATransformer()

# dataset and dataloader
dataset = GraphQADataset(test_tsv=TEST_TSV, tokenizer=tokenizer)

# define collate function
def collate_fn(examples):
    # make container for batches
    batch = {}

    # stack graphs
    batch['node_feat'] = torch.stack([e['node_feat'] for e in examples])
    batch['attn_mask'] = torch.stack([e['attn_mask'] for e in examples])

    # stack text
    batch['input_ids'] = torch.stack([e['input_ids'] for e in examples])
    batch['attention_mask'] = torch.stack([e['attention_mask'] for e in examples])
    batch['labels'] = torch.stack([e['labels'] for e in examples])
    
    return batch

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# training args
training_args = TrainingArguments(
        output_dir="../outputs",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        )

# use hf trainer
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator_fn
        )

if __name__ == '__main__':
    trainer.train()
