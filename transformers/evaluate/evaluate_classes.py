import torch
import numpy as np

# Evaluate model performance - tiny transformer_encoder
def eval_classes_tte(loader, model, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["iner"].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

# Evaluate model performance - transformer_encoder
def eval_classes_te(loader, model, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch["iner"] = batch["iner"].to(device, dtype=torch.float32)
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)