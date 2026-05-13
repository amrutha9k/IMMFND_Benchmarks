import os
import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.optim import AdamW


# The 4 models to train
MODELS_TO_TRAIN = [
    "bert-base-multilingual-cased",      # mBERT
    "google/muril-base-cased",           # MuRIL
    "xlm-roberta-base",                  # XLM-R
    "ai4bharat/IndicBERTv2-MLM-Sam-TLM"  # IndicBERT v2 (TLM variant)
]

EPOCHS = 20
BATCH_SIZE = 32  
LR = 1e-5        
PATIENCE = 5
MAX_LEN = 128    

# Directory mapping 
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Final_Dataset"

# DATASET CREATION
class TextFakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_split(base_dir, split):
    texts, labels = [], []
    
    # Process Fake (Label = 0)
    fake_path = base_dir / split / "Fake" / "Fake_text.xlsx"
    if fake_path.exists():
        df_fake = pd.read_excel(fake_path).dropna(subset=['claim'])
        texts.extend(df_fake['claim'].tolist())
        labels.extend([0] * len(df_fake)) 

    # Process Real (Label = 1)
    real_path = base_dir / split / "Real" / "Real_text.xlsx"
    if real_path.exists():
        df_real = pd.read_excel(real_path).dropna(subset=['claim'])
        texts.extend(df_real['claim'].tolist())
        labels.extend([1] * len(df_real)) 

    return texts, labels


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, acc, f1, prec, rec, auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Load raw text data once
    print("Loading raw Excel data...")
    train_texts, train_labels = load_split(BASE_DIR, "train")
    val_texts, val_labels = load_split(BASE_DIR, "validation")
    test_texts, test_labels = load_split(BASE_DIR, "test")
    print(f"Data Loaded -> Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}\n")

    # List to store final results for the CSV
    final_results = []

    # 2. Loop through all models
    for model_name in MODELS_TO_TRAIN:
        print("="*50)
        print(f"STARTING TRAINING FOR: {model_name}")
        print("="*50)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create Datasets & Loaders specific to this model's tokenizer
        train_loader = DataLoader(TextFakeNewsDataset(train_texts, train_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TextFakeNewsDataset(val_texts, val_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TextFakeNewsDataset(test_texts, test_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)

        # Load Model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        # Optimizer & Scheduler
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

        best_val_f1 = 0.0  
        patience_count = 0
        safe_model_name = model_name.replace("/", "_")
        best_model_path = f"best_model_{safe_model_name}.pt"

        # 3. Training Loop
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch['input_ids'].to(device), 
                    attention_mask=batch['attention_mask'].to(device), 
                    labels=batch['labels'].to(device)
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += outputs.loss.item()

            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = evaluate(model, val_loader, device)

            print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
                print("  -> Checkpoint saved!")
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.\n")
                    break

        # 4. Final Test Evaluation
        print(f"\nEvaluating Best {model_name} on Test Set...")
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, test_f1, test_prec, test_rec, test_auc = evaluate(model, test_loader, device)

        print(f"--- TEST RESULTS FOR {model_name} ---")
        print(f"F1 Macro: {test_f1:.4f} | Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}\n")

        # Save results to list
        final_results.append({
            "Model": model_name,
            "Test Accuracy": round(test_acc, 4),
            "Test F1 (Macro)": round(test_f1, 4),
            "Test Precision": round(test_prec, 4),
            "Test Recall": round(test_rec, 4),
            "Test AUC-ROC": round(test_auc, 4)
        })

        # 5.Clear GPU Memory before loading the next model
        del model, optimizer, scheduler, tokenizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # EXPORT RESULTS TO CSV
    print("-------------")
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    
    results_df = pd.DataFrame(final_results)
    results_csv_path = "model_results_text.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"Summary table saved to: {results_csv_path}")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
