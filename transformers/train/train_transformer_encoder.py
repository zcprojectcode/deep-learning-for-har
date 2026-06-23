import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from .models.transformer_encoder import IMUTransformerEncoder
from .models.callbacks.learning_curves import plot_learning_history

import logging

from sklearn.metrics import confusion_matrix

def train_transformer(train_loader, val_loader, config, device, fold_id, LOG_DIR):
    
    model = IMUTransformerEncoder(config).to(device)

    # Set to train mode
    model.train()

    # Set the optimizer and scheduler
    epochs = config.epochs
    optim = torch.optim.Adam(model.parameters(),
                                lr=config.learning_rate,
                                eps=config.eps,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                step_size=config.step_size,
                                gamma=config.gamma)
    # Set the loss
    loss = torch.nn.NLLLoss()

    n_total_samples = 0.0
    loss_vals = []
    sample_count = []
    logging.info("Start training")
    best_val, best_state, best_epoch = float('inf'), None, 0
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    # Train the transformer encoder model
    for epoch in range(epochs):
        run_loss = 0.0
        train_correct, train_total = 0, 0
        for batch_idx, minibatch in enumerate(train_loader):
            minibatch["iner"] = minibatch["iner"].to(device).to(dtype=torch.float32)
            label = minibatch["label"].to(device, dtype=torch.long)

            optim.zero_grad()
            res = model(minibatch)
            criterion = loss(res, label)
            criterion.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            run_loss += criterion.item()
            train_pred = res.argmax(1) 
            train_correct += (train_pred == label).sum().item()
            train_total += label.numel() 
        
        train_loss = run_loss / len(train_loader)
        train_acc = train_correct / max(1, train_total)
        logging.info(f"Epoch {epoch} Loss: {train_loss}")

        # Validate 
        model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch["iner"] = batch["iner"].to(device, dtype=torch.float32)
                label = batch["label"].to(device, dtype=torch.long)
                res = model(batch)
                total_loss += loss(res, label).item()

                pred = res.argmax(1)
                correct += (pred == label).sum().item()
                total += label.numel()

        val_loss = total_loss / len(val_loader)
        val_acc = correct / max(1, total)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
        model.train()
        scheduler.step()

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

    # Accuracy and loss curves
    plot_learning_history(history=history, path=f"{LOG_DIR}/history{fold_id}.png")

    return best_state, best_val, best_epoch