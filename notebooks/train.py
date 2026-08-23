import os
import torch
import tomli
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from vjeanette.preprocessing import (
    VDJDataset,
    dataset_to_pt
)

from vjeanette.model.model import UNet1D_Embed

pt_path = '/path/to/pt'
test_split = 0.3
val_split = 0.1
batch_size = 256
model_name = 'model_name.pth'
lr = 0.0003
embed_dim = 32
NUM_EPOCHS = 130
early_stopping_patience = 30

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

data = torch.load(pt_path)

X = data["X"]
y = data["y"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_split,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=val_split,
    random_state=42
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=batch_size
)

test_loader = DataLoader(TensorDataset(X, y), batch_size = batch_size, shuffle = True)

import torch.nn.functional as F
def existence_loss(logits, target):
    pred = logits.max(dim=1).values
    tgt = target.max(dim=1).values

    return F.binary_cross_entropy_with_logits(pred, tgt)

def recall_metric(logits, target, thr=0.01):

    probs = torch.sigmoid(logits)

    pred_exist = (probs.max(dim=1).values > thr)
    tgt_exist = (target.max(dim=1).values > 0)

    tp = (pred_exist & tgt_exist).sum().item()
    fn = (~pred_exist & tgt_exist).sum().item()

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)

model = UNet1D_Embed(embed_dim=embed_dim).to(device)

criterion_v = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]).to(device)
)

criterion_j = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]).to(device)
)

criterion_cdr = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]).to(device)
)

optimizer = optim.Adam(
    model.parameters(),
    lr=lr
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=15,
    factor=0.5
)

best_val_loss = float("inf")

early_stopping_counter = 0

def precision_metric(logits, target, thr=0.01):

    probs = torch.sigmoid(logits)

    pred_exist = (probs.max(dim=1).values > thr)
    tgt_exist = (target.max(dim=1).values > 0)

    tp = (pred_exist & tgt_exist).sum().item()
    fp = (pred_exist & ~tgt_exist).sum().item()

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


for epoch in range(NUM_EPOCHS):
        model.train()
        total_v_loss, total_j_loss, total_cdr_loss = 0.0, 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).long()
            batch_y_cdr = batch_y[:,0,:].to(device).float()
            batch_y_v = batch_y[:,1,:].to(device).float()
            batch_y_j = batch_y[:,2,:].to(device).float()
            optimizer.zero_grad()
            outputs_cdr, outputs_v, outputs_j = model(batch_x)
            loss_v = criterion_v(outputs_v, batch_y_v)
            loss_j = criterion_j(outputs_j, batch_y_j)
            loss_cdr = criterion_cdr(outputs_cdr, batch_y_cdr)

        # existence loss
            total_loss = loss_v + loss_j + loss_cdr
            total_loss.backward()
            optimizer.step()

            total_v_loss += loss_v.item()
            total_j_loss += loss_j.item()
            total_cdr_loss += loss_cdr.item()

        # --- Validation ---
        model.eval()
        epoch_val_loss_v, epoch_val_loss_j,  epoch_val_loss_cdr = 0.0, 0.0, 0.0
        recall_v_total, precision_v_total = 0, 0
        recall_j_total, precision_j_total = 0, 0
        recall_cdr_total, precision_cdr_total = 0, 0
        count = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x=val_x.to(device).long()
                val_y_v = val_y[:,1,:].to(device).float()
                val_y_j = val_y[:,2,:].to(device).float()
                val_y_cdr = val_y[:,0,:].to(device).float()
                val_outputs_cdr, val_outputs_v, val_outputs_j = model(val_x)
                loss_v = criterion_v(val_outputs_v, val_y_v)
                loss_j = criterion_j(val_outputs_j, val_y_j)
                loss_cdr = criterion_cdr(val_outputs_cdr, val_y_cdr)
                recall_v_total += recall_metric(val_outputs_v, val_y_v)
                recall_j_total += recall_metric(val_outputs_j, val_y_j)
                recall_cdr_total += recall_metric(val_outputs_cdr, val_y_cdr)
                precision_v_total += precision_metric(val_outputs_v, val_y_v)
                precision_j_total += precision_metric(val_outputs_j, val_y_j)
                precision_cdr_total += precision_metric(val_outputs_cdr, val_y_cdr)
                count+=1
                epoch_val_loss_v += loss_v.item()
                epoch_val_loss_j += loss_j.item()
                epoch_val_loss_cdr += loss_cdr.item()

        train_loss_v = total_v_loss / len(train_loader)
        train_loss_j = total_j_loss / len(train_loader)
        train_loss_cdr = total_cdr_loss / len(train_loader)
        val_loss_v = epoch_val_loss_v / len(val_loader)
        val_loss_j = epoch_val_loss_j / len(val_loader)
        val_loss_cdr = epoch_val_loss_cdr / len(val_loader)

        total_val_loss = val_loss_v + val_loss_j + val_loss_cdr

        # Scheduler step
        scheduler.step(total_val_loss)

        print(f"[{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss - V: {train_loss_v:.4f}, J: {train_loss_j:.4f}, CDR: {train_loss_cdr:.4f}| "
              f"Val Loss - V: {val_loss_v:.4f}, J: {val_loss_j:.4f}, CDR: {val_loss_cdr:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("VAL RECALL V:", recall_v_total/count)
        print("VAL RECALL J:", recall_j_total/count)
        print("VAL RECALL CDR:", recall_cdr_total/count)

        print("VAL PRECISION V:", precision_v_total/count)
        print("VAL PRECISION J:", precision_j_total/count)
        print("VAL PRECISION CDR:", precision_cdr_total/count)


        # --- Early stopping ---
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_weights = model.state_dict().copy()
            early_stopping_counter = 0
            print(f"*** Best model updated ***")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

                
print("VAL RECALL V:", recall_v_total/count)
print("VAL RECALL J:", recall_j_total/count)
print("VAL RECALL CDR:", recall_cdr_total/count)

model.load_state_dict(best_model_weights)

torch.save(
    model.state_dict(),
    model_name
)

print("Training completed.")

print("Best validation loss:", best_val_loss)