import os
import glob
import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPVisionModelWithProjection, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# CONFIGURATION
TEXT_MODELS = [
    "bert-base-multilingual-cased",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
]

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-5        
PATIENCE = 5
MAX_LEN = 128    

# Directory mapping
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Final_Dataset"

class MultimodalCLIPDataset(Dataset):
    def __init__(self, texts, img_paths, labels, tokenizer, clip_processor, max_len):
        self.texts, self.img_paths, self.labels = texts, img_paths, labels
        self.tokenizer, self.clip_processor, self.max_len = tokenizer, clip_processor, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.clip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except Exception: pixel_values = torch.zeros((3, 224, 224))
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'pixel_values': pixel_values, 'labels': torch.tensor(self.labels[idx], dtype=torch.float32)}

def load_multimodal_split(base_dir, split):
    texts, img_paths, labels = [], [], []
    for label_str, label_val in [("Fake", 0), ("Real", 1)]:
        excel_path = base_dir / split / label_str / f"{label_str}_text.xlsx"
        img_folder = base_dir / split / label_str / f"{label_str}_image"
        if not excel_path.exists() or not img_folder.exists(): continue
        df = pd.read_excel(excel_path).dropna(subset=['claim', 'Sr. No'])
        for _, row in df.iterrows():
            claim, sr_no = row['claim'], str(row['Sr. No']).strip()
            found = glob.glob(os.path.join(img_folder, f"{sr_no}.*"))
            if found: texts.append(claim); img_paths.append(found[0]); labels.append(label_val)
    return texts, img_paths, labels

class TextCLIPFusionModel(nn.Module):
    def __init__(self, text_model_name, text_dim=768, img_dim=768, proj_dim=512):
        super().__init__()
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        self.image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(proj_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, input_ids, attention_mask, pixel_values):
        text_features = self.text_proj(self.text_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])
        img_features = self.img_proj(self.image_backbone(pixel_values=pixel_values).image_embeds)
        return self.classifier(torch.cat([text_features, img_features], dim=1)).squeeze(1)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            with torch.cuda.amp.autocast():
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['pixel_values'].to(device))
                loss = criterion(logits, batch['labels'].to(device))
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs); all_preds.extend((probs > 0.5).astype(int)); all_labels.extend(batch['labels'].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds); f1 = f1_score(all_labels, all_preds, average='macro'); prec = precision_score(all_labels, all_preds, average='macro', zero_division=0); rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.0
    return total_loss / len(loader), acc, f1, prec, rec, auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading image-text pairs...")
    tr_txt, tr_img, tr_lbl = load_multimodal_split(BASE_DIR, "train")
    v_txt, v_img, v_lbl = load_multimodal_split(BASE_DIR, "validation")
    te_txt, te_img, te_lbl = load_multimodal_split(BASE_DIR, "test")

    final_results = []
    criterion = nn.BCEWithLogitsLoss()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    for text_model in TEXT_MODELS:
        print("\n" + "="*50)
        print(f"PROCESSING MODEL: {text_model} + CLIP (ViT-Large)")
        print("="*50)

        safe_name = text_model.replace("/", "_")
        best_model_path = f"best_clip_{safe_name}_0.3.pt"
        
        # --- SKIP IF ALREADY TRAINED ---
        if os.path.exists(best_model_path):
            print(f"Checkpoint found for {text_model}. Loading results directly.")
            tokenizer = AutoTokenizer.from_pretrained(text_model)
            test_loader = DataLoader(MultimodalCLIPDataset(te_txt, te_img, te_lbl, tokenizer, clip_processor, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
            model = TextCLIPFusionModel(text_model).to(device)
            model.load_state_dict(torch.load(best_model_path))
            _, t_acc, t_f1, t_prec, t_rec, t_auc = evaluate(model, test_loader, criterion, device)
        else:
            print(f"No checkpoint found. Starting training for {text_model}...")
            tokenizer = AutoTokenizer.from_pretrained(text_model)
            
            train_loader = DataLoader(MultimodalCLIPDataset(tr_txt, tr_img, tr_lbl, tokenizer, clip_processor, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
            val_loader = DataLoader(MultimodalCLIPDataset(v_txt, v_img, v_lbl, tokenizer, clip_processor, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
            test_loader = DataLoader(MultimodalCLIPDataset(te_txt, te_img, te_lbl, tokenizer, clip_processor, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

            model = TextCLIPFusionModel(text_model).to(device)
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * len(train_loader) * EPOCHS), num_training_steps=len(train_loader) * EPOCHS)
            scaler = torch.cuda.amp.GradScaler()

            best_val_f1 = 0.0  
            patience_count = 0

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for batch in tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False):
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['pixel_values'].to(device))
                        loss = criterion(logits, batch['labels'].to(device))
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); scheduler.step()

                _, _, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_count = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE: break
            
            print(f"Testing {text_model} + CLIP...")
            model.load_state_dict(torch.load(best_model_path))
            _, t_acc, t_f1, t_prec, t_rec, t_auc = evaluate(model, test_loader, criterion, device)
        
        final_results.append({"Model": f"{text_model} + CLIP", "Accuracy": t_acc, "F1 Macro": t_f1, "Precision": t_prec, "Recall": t_rec, "AUC": t_auc})
        
        # --- SAVE CSV INCREMENTALLY ---
        pd.DataFrame(final_results).to_csv("clip_multimodal_results_0.3.csv", index=False)
        print(f"Results for {text_model} saved. CSV is up to date.")

        del model, tokenizer
        torch.cuda.empty_cache(); gc.collect()

    print("\nAll CLIP models processed! Final results are in clip_multimodal_results_0.3.csv")

if __name__ == "__main__":
    main()
