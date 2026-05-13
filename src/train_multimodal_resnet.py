import os
import glob
import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from tqdm import tqdm
from torch.optim import AdamW
from pathlib import Path
from PIL import Image


# CONFIGURATION
TEXT_MODELS = [
    "bert-base-multilingual-cased",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
]

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-5        
PATIENCE = 5
MAX_LEN = 128    

# Directory mapping
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Final_Dataset"

# Standard ResNet Preprocessing
res_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DATASET CREATION
class MultimodalResNetDataset(Dataset):
    def __init__(self, texts, img_paths, labels, tokenizer, max_len, transform):
        self.texts = texts
        self.img_paths = img_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )

        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            image = torch.zeros((3, 224, 224))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32) 
        }

def load_multimodal_split(base_dir, split):
    texts, img_paths, labels = [], [], []
    for label_str, label_val in [("Fake", 0), ("Real", 1)]:
        excel_path = base_dir / split / label_str / f"{label_str}_text.xlsx"
        img_folder = base_dir / split / label_str / f"{label_str}_image"
        
        if not excel_path.exists() or not img_folder.exists(): continue

        df = pd.read_excel(excel_path).dropna(subset=['claim', 'Sr. No'])
        for _, row in df.iterrows():
            claim = row['claim']
            sr_no = str(row['Sr. No']).strip()
            found_images = glob.glob(os.path.join(img_folder, f"{sr_no}.*"))
            if found_images:
                texts.append(claim)
                img_paths.append(found_images[0])
                labels.append(label_val)
    return texts, img_paths, labels


# MULTIMODAL MODEL ARCHITECTURE
class TextResNetFusionModel(nn.Module):
    def __init__(self, text_model_name, text_dim=768, img_dim=2048, proj_dim=512):
        super().__init__()
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        
        resnet = models.resnet50(pretrained=True)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1]) # 2048 dims
        
        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, images):
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_cls)
        
        img_outputs = self.image_backbone(images)
        img_features = self.img_proj(torch.flatten(img_outputs, 1))
        
        fused = torch.cat([text_features, img_features], dim=1)
        return self.classifier(fused).squeeze(1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.0

    return avg_loss, acc, f1, prec, rec, auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading image-text pairs...")
    tr_txt, tr_img, tr_lbl = load_multimodal_split(BASE_DIR, "train")
    v_txt, v_img, v_lbl = load_multimodal_split(BASE_DIR, "validation")
    te_txt, te_img, te_lbl = load_multimodal_split(BASE_DIR, "test")

    final_results = []
    criterion = nn.BCEWithLogitsLoss()

    for text_model in TEXT_MODELS:
        print("\n" + "="*50)
        print(f"TRAINING: {text_model} + ResNet-50")
        print("="*50)

        tokenizer = AutoTokenizer.from_pretrained(text_model)
        
        train_loader = DataLoader(MultimodalResNetDataset(tr_txt, tr_img, tr_lbl, tokenizer, MAX_LEN, res_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=16,pin_memory=True)
        val_loader = DataLoader(MultimodalResNetDataset(v_txt, v_img, v_lbl, tokenizer, MAX_LEN, res_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(MultimodalResNetDataset(te_txt, te_img, te_lbl, tokenizer, MAX_LEN, res_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

        model = TextResNetFusionModel(text_model).to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * len(train_loader) * EPOCHS), num_training_steps=len(train_loader) * EPOCHS)
        scaler = torch.cuda.amp.GradScaler()

        best_val_f1 = 0.0  
        patience_count = 0
        safe_name = text_model.replace("/", "_")
        best_model_path = f"best_resnet_{safe_name}_0.3.pt"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device))
                    loss = criterion(logits, batch['labels'].to(device))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:02d} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_count += 1
                if patience_count >= PATIENCE: break

        print(f"Testing {text_model} + ResNet-50...")
        model.load_state_dict(torch.load(best_model_path))
        _, t_acc, t_f1, t_prec, t_rec, t_auc = evaluate(model, test_loader, criterion, device)
        
        final_results.append({"Model": f"{text_model} + ResNet50", "Accuracy": t_acc, "F1 Macro": t_f1, "Precision": t_prec, "Recall": t_rec, "AUC": t_auc})
        
        del model, optimizer, scheduler, tokenizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(final_results).to_csv("resnet_multimodal_results_0.3.csv", index=False)
    print("All ResNet models trained! Results saved to resnet_multimodal_results_0.3.csv")

if __name__ == "__main__":
    main()
