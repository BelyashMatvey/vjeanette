# train.py

import os
import torch
import tomli
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

from vjeanette.preprocessing import VDJDataset, dataset_to_pt
from vjeanette.model.model import UNet1D_Embed


def load_config():
    with open("pyproject.toml", "rb") as f:
        return tomli.load(f)["tool"]["vjeanette"]
def existence_loss(logits, target):
    """
    logits: [B, L]
    target: [B, L]
    """
    pred = torch.sigmoid(logits).max(dim=1).values
    tgt = target.max(dim=1).values
    return nn.BCELoss()(pred, tgt)
def recall_metric(logits, target, thr=0.01):
    probs = torch.sigmoid(logits)
    pred_exist = (probs.max(dim=1).values > thr)
    tgt_exist = (target.max(dim=1).values > 0)

    tp = (pred_exist & tgt_exist).sum().item()
    fn = (~pred_exist & tgt_exist).sum().item()

    if tp+fn == 0:
        return 0
    return tp/(tp+fn)


def main():
    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- DATA ----------
    pt_path = cfg["training"]["pt_output"]

    if not os.path.exists(pt_path):
        print("Creating dataset...")
        ds = VDJDataset(
            ig_path=cfg["training"]["ig_file"],
            tcr_path=cfg["training"]["tcr_file"],
            neg_ratio=cfg["training"]["neg_ratio"]
        )
        dataset_to_pt(ds, pt_path)

    data = torch.load(pt_path)
    X, y = data["X"], data["y"]

    # ---------- SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["training"]["test_split"], random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=cfg["training"]["val_split"], random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg["training"]["batch_size"]
    )
    model = UNet1D_Embed(embed_dim=32).to(device)
    criterion_v = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device)) 
    criterion_j = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device)) 
    criterion_cdr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device)) 
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    early_stopping_patience = 30
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model_weights = None
    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_v_loss, total_j_loss, total_cdr_loss = 0.0, 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).long()
            batch_y_cdr = batch_y[:,0,:].to(device).float()
            batch_y_v = batch_y[:,1,:].to(device).float()
            batch_y_j = batch_y[:,2,:].to(device).float()
            optimizer.zero_grad()
            outputs_v, outputs_j, outputs_cdr = model(batch_x)
            loss_v = criterion_v(outputs_v, batch_y_v)
            loss_j = criterion_j(outputs_j, batch_y_j)
            loss_cdr = criterion_cdr(outputs_cdr, batch_y_cdr)

        # existence loss
            loss_v_exist = existence_loss(outputs_v, batch_y_v)
            loss_j_exist = existence_loss(outputs_j, batch_y_j)
            loss_cdr_exist = existence_loss(outputs_cdr, batch_y_cdr)
            total_loss = (
                loss_v + loss_j + loss_cdr
                + 2.0 * (loss_v_exist + loss_j_exist + loss_cdr_exist)  # <-- ключ
            )
            total_loss.backward()
            optimizer.step()

            total_v_loss += loss_v.item()
            total_j_loss += loss_j.item()
            total_cdr_loss += loss_cdr.item()

        # --- Validation ---
        model.eval()
        epoch_val_loss_v, epoch_val_loss_j,  epoch_val_loss_cdr = 0.0, 0.0, 0.0
        recall_v_total = 0
        recall_j_total = 0
        recall_cdr_total = 0
        count = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x=val_x.to(device).long()
                val_y_v = val_y[:,1,:].to(device).float()
                val_y_j = val_y[:,2,:].to(device).float()
                val_y_cdr = val_y[:,0,:].to(device).float()
                val_outputs_v, val_outputs_j, val_outputs_cdr = model(val_x)
                loss_v = criterion_v(val_outputs_v, val_y_v)
                loss_j = criterion_j(val_outputs_j, val_y_j)
                loss_cdr = criterion_cdr(val_outputs_cdr, val_y_cdr)
                recall_v_total += recall_metric(val_outputs_v, val_y_v)
                recall_j_total += recall_metric(val_outputs_j, val_y_j)
                recall_cdr_total += recall_metric(val_outputs_cdr, val_y_cdr)
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

        print(f"[{epoch+1}/{cfg['training']['epochs']}] "
              f"Train Loss - V: {train_loss_v:.4f}, J: {train_loss_j:.4f}, CDR: {train_loss_cdr:.4f}| "
              f"Val Loss - V: {val_loss_v:.4f}, J: {val_loss_j:.4f}, CDR: {val_loss_cdr:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("VAL RECALL V:", recall_v_total/count)
        print("VAL RECALL J:", recall_j_total/count)
        print("VAL RECALL CDR:", recall_cdr_total/count)


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

    # Восстановление лучших весов
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), cfg['training']['model_name'])
    print("Training completed. Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()