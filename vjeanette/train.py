#!/usr/bin/env python
# coding: utf-8

"""
Training script for VDJ sequence model.

This script trains a UNet1D_Embed model on VDJ sequence data
for predicting CDR3, V, and J gene positions.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from vjeanette.preprocessing import VDJDataset, dataset_to_pt
from vjeanette.model.model import UNet1D_Embed


def existence_loss(logits, target):
    """
    Calculate existence loss based on maximum values.
    
    Args:
        logits: Model output logits
        target: Target binary masks
    
    Returns:
        Binary cross-entropy loss for existence prediction
    """
    pred = logits.max(dim=1).values
    tgt = target.max(dim=1).values
    return F.binary_cross_entropy_with_logits(pred, tgt)


def recall_metric(logits, target, thr=0.01):
    """
    Calculate recall metric for sequence prediction.
    
    Args:
        logits: Model output logits
        target: Target binary masks
        thr: Probability threshold for positive prediction
    
    Returns:
        Recall score (true positives / (true positives + false negatives))
    """
    probs = torch.sigmoid(logits)
    pred_exist = probs.max(dim=1).values > thr
    tgt_exist = target.max(dim=1).values > 0

    tp = (pred_exist & tgt_exist).sum().item()
    fn = (~pred_exist & tgt_exist).sum().item()

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def precision_metric(logits, target, thr=0.01):
    """
    Calculate precision metric for sequence prediction.
    
    Args:
        logits: Model output logits
        target: Target binary masks
        thr: Probability threshold for positive prediction
    
    Returns:
        Precision score (true positives / (true positives + false positives))
    """
    probs = torch.sigmoid(logits)
    pred_exist = probs.max(dim=1).values > thr
    tgt_exist = target.max(dim=1).values > 0

    tp = (pred_exist & tgt_exist).sum().item()
    fp = (pred_exist & ~tgt_exist).sum().item()

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def load_data(pt_path, test_split=0.3, val_split=0.1, batch_size=256, 
              random_state=42, verbose=True):
    """
    Load and split dataset into train, validation, and test sets.
    
    Args:
        pt_path: Path to serialized dataset file
        test_split: Fraction of data for test set
        val_split: Fraction of training data for validation
        batch_size: Batch size for DataLoader
        random_state: Random seed for reproducibility
        verbose: Print progress information
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if verbose:
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        print(f"Loading dataset from: {pt_path}")

    data = torch.load(pt_path)
    X, y = data["X"], data["y"]

    if verbose:
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )

    # Split train into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=random_state
    )

    if verbose:
        print(f"Train samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Test samples: {len(X_test):,}")

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    if verbose:
        print("=" * 60)
        print()

    return train_loader, val_loader, test_loader


def create_model(embed_dim=32, device="cuda", verbose=True):
    """
    Create and initialize the UNet model.
    
    Args:
        embed_dim: Embedding dimension
        device: Device to place model on
        verbose: Print progress information
    
    Returns:
        Initialized model
    """
    if verbose:
        print("Creating model...")
        print(f"Embedding dimension: {embed_dim}")

    model = UNet1D_Embed(embed_dim=embed_dim).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print()

    return model


def create_optimizer(model, lr=0.0003, pos_weight=2.0, patience=15, 
                     factor=0.5, verbose=True):
    """
    Create optimizer, scheduler, and loss functions.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        pos_weight: Positive class weight for BCE loss
        patience: Patience for learning rate scheduler
        factor: Factor for learning rate reduction
        verbose: Print progress information
    
    Returns:
        Tuple of (criterion_v, criterion_j, criterion_cdr, optimizer, scheduler)
    """
    if verbose:
        print("Setting up optimizer and loss functions...")

    # Loss functions with positive class weighting
    criterion_v = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(model.device if hasattr(model, 'device') else 'cuda')
    )
    criterion_j = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(model.device if hasattr(model, 'device') else 'cuda')
    )
    criterion_cdr = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(model.device if hasattr(model, 'device') else 'cuda')
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=patience,
        factor=factor
    )

    if verbose:
        print(f"Learning rate: {lr}")
        print(f"Positive weight: {pos_weight}")
        print(f"Scheduler patience: {patience}")
        print(f"Scheduler factor: {factor}")
        print()

    return criterion_v, criterion_j, criterion_cdr, optimizer, scheduler


def train_epoch(model, train_loader, criterion_v, criterion_j, criterion_cdr, 
                optimizer, device, verbose=False):
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion_v: V gene loss function
        criterion_j: J gene loss function
        criterion_cdr: CDR3 loss function
        optimizer: Optimizer
        device: Device for computation
        verbose: Print progress information
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    total_v_loss, total_j_loss, total_cdr_loss = 0.0, 0.0, 0.0
    total_loss = 0.0

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        # Move data to device
        batch_x = batch_x.to(device).long()
        batch_y_cdr = batch_y[:, 0, :].to(device).float()
        batch_y_v = batch_y[:, 1, :].to(device).float()
        batch_y_j = batch_y[:, 2, :].to(device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs_cdr, outputs_v, outputs_j = model(batch_x)

        # Calculate losses
        loss_v = criterion_v(outputs_v, batch_y_v)
        loss_j = criterion_j(outputs_j, batch_y_j)
        loss_cdr = criterion_cdr(outputs_cdr, batch_y_cdr)
        batch_loss = loss_v + loss_j + loss_cdr

        # Backward pass
        batch_loss.backward()
        optimizer.step()

        # Accumulate losses
        total_v_loss += loss_v.item()
        total_j_loss += loss_j.item()
        total_cdr_loss += loss_cdr.item()
        total_loss += batch_loss.item()

        if verbose and batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={batch_loss.item():.4f}")

    # Calculate average losses
    num_batches = len(train_loader)
    avg_losses = {
        'v': total_v_loss / num_batches,
        'j': total_j_loss / num_batches,
        'cdr': total_cdr_loss / num_batches,
        'total': total_loss / num_batches
    }

    return avg_losses


def validate(model, val_loader, criterion_v, criterion_j, criterion_cdr, 
             device, verbose=False):
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion_v: V gene loss function
        criterion_j: J gene loss function
        criterion_cdr: CDR3 loss function
        device: Device for computation
        verbose: Print progress information
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    epoch_val_loss_v, epoch_val_loss_j, epoch_val_loss_cdr = 0.0, 0.0, 0.0
    total_val_loss = 0.0
    
    recall_v_total, precision_v_total = 0.0, 0.0
    recall_j_total, precision_j_total = 0.0, 0.0
    recall_cdr_total, precision_cdr_total = 0.0, 0.0
    
    num_batches = 0

    with torch.no_grad():
        for val_x, val_y in val_loader:
            # Move data to device
            val_x = val_x.to(device).long()
            val_y_v = val_y[:, 1, :].to(device).float()
            val_y_j = val_y[:, 2, :].to(device).float()
            val_y_cdr = val_y[:, 0, :].to(device).float()

            # Forward pass
            val_outputs_cdr, val_outputs_v, val_outputs_j = model(val_x)

            # Calculate losses
            loss_v = criterion_v(val_outputs_v, val_y_v)
            loss_j = criterion_j(val_outputs_j, val_y_j)
            loss_cdr = criterion_cdr(val_outputs_cdr, val_y_cdr)

            # Accumulate losses
            epoch_val_loss_v += loss_v.item()
            epoch_val_loss_j += loss_j.item()
            epoch_val_loss_cdr += loss_cdr.item()
            total_val_loss += (loss_v.item() + loss_j.item() + loss_cdr.item())

            # Calculate metrics
            recall_v_total += recall_metric(val_outputs_v, val_y_v)
            recall_j_total += recall_metric(val_outputs_j, val_y_j)
            recall_cdr_total += recall_metric(val_outputs_cdr, val_y_cdr)
            
            precision_v_total += precision_metric(val_outputs_v, val_y_v)
            precision_j_total += precision_metric(val_outputs_j, val_y_j)
            precision_cdr_total += precision_metric(val_outputs_cdr, val_y_cdr)

            num_batches += 1

    # Calculate averages
    metrics = {
        'loss': {
            'v': epoch_val_loss_v / num_batches,
            'j': epoch_val_loss_j / num_batches,
            'cdr': epoch_val_loss_cdr / num_batches,
            'total': total_val_loss / num_batches
        },
        'recall': {
            'v': recall_v_total / num_batches,
            'j': recall_j_total / num_batches,
            'cdr': recall_cdr_total / num_batches
        },
        'precision': {
            'v': precision_v_total / num_batches,
            'j': precision_j_total / num_batches,
            'cdr': precision_cdr_total / num_batches
        }
    }

    return metrics


def train_model(model, train_loader, val_loader, criterion_v, criterion_j, 
                criterion_cdr, optimizer, scheduler, device, num_epochs=130, 
                early_stopping_patience=30, verbose=True):
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion_v: V gene loss function
        criterion_j: J gene loss function
        criterion_cdr: CDR3 loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device for computation
        num_epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        verbose: Print progress information
    
    Returns:
        Tuple of (best_val_loss, best_model_weights)
    """
    if verbose:
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

    best_val_loss = float("inf")
    best_model_weights = None
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        train_losses = train_epoch(
            model, train_loader, criterion_v, criterion_j, criterion_cdr, 
            optimizer, device, verbose=False
        )

        # Validation phase
        val_metrics = validate(
            model, val_loader, criterion_v, criterion_j, criterion_cdr, 
            device, verbose=False
        )

        # Update learning rate
        scheduler.step(val_metrics['loss']['total'])

        # Print progress
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss - V: {train_losses['v']:.4f}, "
                  f"J: {train_losses['j']:.4f}, "
                  f"CDR: {train_losses['cdr']:.4f}")
            print(f"  Val Loss - V: {val_metrics['loss']['v']:.4f}, "
                  f"J: {val_metrics['loss']['j']:.4f}, "
                  f"CDR: {val_metrics['loss']['cdr']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Recall - V: {val_metrics['recall']['v']:.4f}, "
                  f"J: {val_metrics['recall']['j']:.4f}, "
                  f"CDR: {val_metrics['recall']['cdr']:.4f}")
            print(f"  Precision - V: {val_metrics['precision']['v']:.4f}, "
                  f"J: {val_metrics['precision']['j']:.4f}, "
                  f"CDR: {val_metrics['precision']['cdr']:.4f}")

        # Early stopping check
        if val_metrics['loss']['total'] < best_val_loss:
            best_val_loss = val_metrics['loss']['total']
            best_model_weights = model.state_dict().copy()
            early_stopping_counter = 0
            if verbose:
                print("  *** Best model updated ***")
        else:
            early_stopping_counter += 1
            if verbose:
                print(f"  Early stopping counter: "
                      f"{early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if verbose:
            print()

    return best_val_loss, best_model_weights


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train VDJ sequence prediction model"
    )
    
    parser.add_argument(
        "--pt_path",
        type=str,
        required=True,
        help="Path to preprocessed dataset (.pt file)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="model.pth",
        help="Output path for trained model (default: model.pth)"
    )
    
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.3,
        help="Fraction of data for test set (default: 0.3)"
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data for validation (default: 0.1)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate (default: 0.0003)"
    )
    
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=32,
        help="Embedding dimension (default: 32)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=130,
        help="Maximum number of epochs (default: 130)"
    )
    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Patience for early stopping (default: 30)"
    )
    
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=2.0,
        help="Positive class weight for BCE loss (default: 2.0)"
    )
    
    args = parser.parse_args()

    # Setup device
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load and split data
    train_loader, val_loader, test_loader = load_data(
        pt_path=args.pt_path,
        test_split=args.test_split,
        val_split=args.val_split,
        batch_size=args.batch_size
    )

    # Create model
    model = create_model(
        embed_dim=args.embed_dim,
        device=device
    )

    # Setup optimizer and loss functions
    criterion_v, criterion_j, criterion_cdr, optimizer, scheduler = create_optimizer(
        model=model,
        lr=args.lr,
        pos_weight=args.pos_weight
    )

    # Train model
    best_val_loss, best_model_weights = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_v=criterion_v,
        criterion_j=criterion_j,
        criterion_cdr=criterion_cdr,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )

    # Save best model
    print("=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), args.model_name)
    
    print(f"Model saved to: {args.model_name}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()