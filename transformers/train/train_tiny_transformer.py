"""
Lamaakal, I., Yahyati, C., Maleh, Y. et al. 
A tiny inertial transformer for human activity recognition via multimodal knowledge 
distillation and explainable AI. Sci Rep 15, 42335 (2025). 
https://doi.org/10.1038/s41598-025-26297-2

MIT License

Copyright (c) 2025 Ismail Lamaakal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/Ism-ail11/XTinyHAR/tree/main?tab=MIT-1-ov-file
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from .models.tiny_transformer import TinyTransformer
from .models.callbacks.learning_curves import plot_learning_history

import logging

from sklearn.metrics import confusion_matrix

# Tiny transformer model training
def train_tiny_transformer(train_loader, val_loader, config, fold_id, LOG_DIR):
    
    student = TinyTransformer(
        in_channels=config.features, 
        window_size=config.window, 
        patch_size=config.patch,
        dim=config.dimension, 
        depth=config.depth, 
        heads=config.heads, 
        mlp_ratio=config.mlp_ratio,
        num_classes=config.classes, 
        drop=config.drop).to(config.device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        student.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay)

    best_val = -1.0
    best_state = None

    best_labels = []
    best_preds = []
    best_epoch = -1

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for ep in range(config.epochs):
        student.train()
        pbar = tqdm(train_loader, desc=f"Train [{ep+1}/{config.epochs}]")
        run_loss = 0.0
        train_correct, train_total = 0, 0

        for batch in pbar:
            x_iner = batch["iner"].to(config.device)
            y = batch["label"].to(config.device)

            opt.zero_grad(set_to_none=True)
            s_logits, _ = student(x_iner)

            loss = criterion(s_logits, y)
            loss.backward()
            opt.step()
            
            run_loss += loss.item()
            train_pred = s_logits.argmax(1)
            train_correct += (train_pred == y).sum().item()
            train_total += y.numel()
            pbar.set_postfix(loss=run_loss / (pbar.n+1))

        train_loss = run_loss / len(train_loader)
        train_acc = train_correct / max(1, train_total)
        
        student.eval()
        correct, total = 0, 0
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["iner"].to(config.device)
                y = batch["label"]

                logits, _ = student(batch["iner"].to(config.device))
                pred = logits.argmax(1).cpu()

                # Loss and accuracy
                val_loss += criterion(logits, batch["label"].to(config.device)).item()
                correct += (pred == batch["label"]).sum().item()
                total += pred.numel()

                # Store for confusion matrix
                all_preds.append(pred)
                all_labels.append(y)
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / max(1,total)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
            best_preds = all_preds
            best_labels = all_labels
            best_epoch = ep
        
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
    
    # Accuracy and loss curves
    plot_learning_history(history=history, path=f"{LOG_DIR}/history{fold_id}.png")

    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds, normalize='true')

    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    logging.info("\nPer-class accuracy:")
    for i, acc in enumerate(class_acc):
        logging.info(f"Class {i}: {acc:.4f}")

    return best_state, best_val, best_epoch