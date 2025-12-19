#!/usr/bin/env python
"""Train the unified mood classification model with multitask learning"""

import argparse
from contextlib import nullcontext
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from src.models.unified_model import UnifiedMoodModel
from src.losses.multitask_loss import MultitaskLoss
from dataloader_fast import get_dataloader_fast

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable


def _binarize_with_top1_fallback(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize probabilities with a global threshold and top-1 fallback.

    For each sample, if no class exceeds threshold, the argmax class is forced to 1.

    y_score: [N, C] in [0,1]
    returns y_pred: [N, C] in {0,1}
    """
    if y_score.ndim != 2:
        raise ValueError(f"y_score must be 2D [N,C], got shape={y_score.shape}")

    y_pred = (y_score >= float(threshold)).astype(np.int32)
    empty = y_pred.sum(axis=1) == 0
    if np.any(empty):
        top1 = np.argmax(y_score[empty], axis=1)
        y_pred[empty, top1] = 1
    return y_pred


def _search_global_threshold_f1_micro(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the global threshold that maximizes F1-micro with top-1 fallback.

    Returns (best_threshold, best_f1_micro).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    thresholds = np.asarray(thresholds, dtype=np.float32)

    best_t = float(thresholds[0])
    best_f1 = -1.0

    for t in thresholds:
        y_pred = _binarize_with_top1_fallback(y_score, float(t))
        f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def _ccc_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    # pred/target: [N, D]
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_var = ((pred - pred_mean) ** 2).mean(axis=0)
    target_var = ((target - target_mean) ** 2).mean(axis=0)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)
    return float(np.nanmean(ccc))


def _pearson_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    # pred/target: [N, D]
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    cov = (pred_centered * target_centered).mean(axis=0)
    pred_std = np.sqrt((pred_centered**2).mean(axis=0))
    target_std = np.sqrt((target_centered**2).mean(axis=0))
    r = cov / (pred_std * target_std + eps)
    return float(np.nanmean(r))


def _rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    # pred/target: [N, D]
    rmse_per_dim = np.sqrt(((pred - target) ** 2).mean(axis=0))
    return float(np.nanmean(rmse_per_dim))


def _mtg_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute multilabel metrics.

    y_true: [N, C] in {0,1}
    y_score: [N, C] in [0,1]
    """
    metrics: dict[str, float] = {}
    y_pred = _binarize_with_top1_fallback(y_score=y_score, threshold=threshold)

    # F1
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Precision / Recall
    metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC
    try:
        metrics["roc_auc_micro"] = float(roc_auc_score(y_true, y_score, average="micro"))
    except ValueError:
        metrics["roc_auc_micro"] = float("nan")

    per_class_roc = []
    for c in range(y_true.shape[1]):
        # ROC-AUC undefined if the class is all-0 or all-1
        if np.unique(y_true[:, c]).size < 2:
            continue
        try:
            per_class_roc.append(float(roc_auc_score(y_true[:, c], y_score[:, c])))
        except ValueError:
            continue
    metrics["roc_auc_macro"] = float(np.mean(per_class_roc)) if per_class_roc else float("nan")

    # PR-AUC (Average Precision)
    try:
        metrics["pr_auc_micro"] = float(average_precision_score(y_true, y_score, average="micro"))
    except ValueError:
        metrics["pr_auc_micro"] = float("nan")

    try:
        metrics["pr_auc_macro"] = float(average_precision_score(y_true, y_score, average="macro"))
    except ValueError:
        # Fall back to manual macro that skips undefined classes (no positive samples)
        per_class_ap = []
        for c in range(y_true.shape[1]):
            if y_true[:, c].sum() == 0:
                continue
            try:
                per_class_ap.append(float(average_precision_score(y_true[:, c], y_score[:, c])))
            except ValueError:
                continue
        metrics["pr_auc_macro"] = float(np.mean(per_class_ap)) if per_class_ap else float("nan")

    return metrics


def _print_dataset_summaries(deam_loader, mtg_train_loader, mtg_val_loader) -> None:
    """Print one-time dataset label summaries to verify scales."""
    # DEAM label scale sanity check
    deam_ds = getattr(deam_loader, "dataset", None)
    deam_labels_dict = getattr(deam_ds, "labels", None)
    if isinstance(deam_labels_dict, dict) and deam_labels_dict:
        va = np.asarray(list(deam_labels_dict.values()), dtype=np.float32)  # [N, 2]
        v = va[:, 0]
        a = va[:, 1]
        print(
            "DEAM label range (static song-level): "
            f"valence min/max={v.min():.3f}/{v.max():.3f}, "
            f"arousal min/max={a.min():.3f}/{a.max():.3f}"
        )

    # MTG label sparsity sanity check (based on loaded split, after audio existence filtering)
    for split_name, loader in [("train", mtg_train_loader), ("val", mtg_val_loader)]:
        ds = getattr(loader, "dataset", None)
        mood_tags = getattr(ds, "mood_tags", None)
        data = getattr(ds, "data", None)
        if isinstance(mood_tags, list) and isinstance(data, list) and mood_tags and data:
            pos_per_item = np.asarray([len(item.get("mood_tags", [])) for item in data], dtype=np.float32)
            density = float(pos_per_item.mean() / len(mood_tags))
            print(
                f"MTG-{split_name} tags: C={len(mood_tags)} "
                f"avg_pos_per_sample={pos_per_item.mean():.3f} "
                f"(density~{density:.4f})"
            )

def main():
    # 1. 参数解析
    parser = argparse.ArgumentParser(description="Train the unified mood classification model")
    parser.add_argument('--fusion_type', type=str, default='early', choices=['early', 'late'], help='Feature fusion type')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training') 
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--deam_weight', type=float, default=0.5, help='Weight for DEAM regression task')
    parser.add_argument('--mtg_weight', type=float, default=0.5, help='Weight for MTG-Jamendo classification task')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--train_pct', type=float, default=1.0, help='Use only this fraction of training data (0-1]')
    parser.add_argument('--subset_seed', type=int, default=42, help='Seed for dataset sub-sampling')
    parser.add_argument('--no_progress', action='store_true', help='Disable tqdm progress bars')
    parser.add_argument('--mtg_debug', action='store_true', help='Print extra MTG debug info during validation')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--log_interval', type=int, default=10, help='Log training metrics every N optimizer steps')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers for precomputed features (suggest 2-6 on Mac)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for model optimization (requires PyTorch 2.0+)')

    args = parser.parse_args()

    # 2. 设备配置（mac 优先使用 MPS）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = device.type == "mps"
    autocast_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16) if use_amp else nullcontext()
    
    # 3. 数据加载
    # 始终使用快速模式（预计算特征）。
    # 训练前请先运行：python precompute_features.py --dataset all
    print("\n🚀 使用快速模式 (预计算特征)")
    loader_fn = get_dataloader_fast
    loader_kwargs = {'num_workers': args.num_workers}

    # Load DEAM datasets (train/val/test)
    print("Loading DEAM dataset...")
    deam_train_loader = loader_fn(
        "deam",
        split="train",
        batch_size=args.batch_size,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
        **loader_kwargs,
    )
    deam_val_loader = loader_fn(
        "deam",
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
        **loader_kwargs,
    )
    deam_test_loader = loader_fn(
        "deam",
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
        **loader_kwargs,
    )

    # Load MTG-Jamendo datasets (train/val/test)
    print("Loading MTG-Jamendo dataset...")
    mtg_train_loader = loader_fn(
        "mtg-jamendo",
        split="train",
        batch_size=args.batch_size,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
        **loader_kwargs,
    )
    mtg_val_loader = loader_fn(
        "mtg-jamendo",
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
        **loader_kwargs,
    )
    mtg_test_loader = loader_fn(
        "mtg-jamendo",
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        subset_fraction=args.train_pct,
        subset_seed=args.subset_seed,
        **loader_kwargs,
    )

    num_mtg_tags = len(getattr(mtg_train_loader.dataset, "mood_tags", []))
    if num_mtg_tags <= 0:
        raise RuntimeError("MTG dataset has no mood tags; cannot determine classification head size")

    _print_dataset_summaries(deam_train_loader, mtg_train_loader, mtg_val_loader)

    # 4. 模型初始化
    print("Initializing model...")
    model = UnifiedMoodModel(
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        num_class_tags=num_mtg_tags
    )
    model = model.to(device)  # Use full precision, let AMP handle automatic mixed precision

    # 可选: 使用 torch.compile 加速 (PyTorch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            # 注意: MPS 后端对 torch.compile 支持有限，使用 'reduce-overhead' 模式
            model = torch.compile(model, mode='reduce-overhead')
            print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️ torch.compile failed, using eager mode: {e}")

    # 5. 损失函数与优化器
    loss_fn = MultitaskLoss().to(device)  # Move loss to device
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 6. TensorBoard 初始化
    log_dir = f"runs/unified_mood_model_{args.fusion_type}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Track best MTG threshold (selected on validation set) across epochs
    best_mtg_val_f1_micro = -1.0
    best_mtg_threshold = None
    best_mtg_epoch = None
    best_val_loss = float('inf')
    start_epoch = 0
    optimizer_step = 0

    # 6.1 Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded.")

            # Load optimizer state (if available)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded.")
                # Override learning rate with current args.lr (allows tuning LR on resume)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                print(f"Learning rate set to: {args.lr}")
            else:
                print("Warning: No optimizer state in checkpoint, optimizer starts fresh.")

            # Restore epoch counter
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch + 1}")

            # Restore step counter (for TensorBoard alignment)
            optimizer_step = int(checkpoint.get('optimizer_step', 0))

            # Restore best metrics trackers
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_mtg_val_f1_micro = checkpoint.get('best_mtg_val_f1_micro', -1.0)
            best_mtg_threshold = checkpoint.get('best_mtg_threshold', None)
            best_mtg_epoch = checkpoint.get('best_mtg_epoch', None)
            print(f"Restored best_val_loss={best_val_loss:.4f}, best_mtg_f1_micro={best_mtg_val_f1_micro:.4f}")
        else:
            print(f"Warning: Checkpoint not found at '{args.resume}', starting from scratch.")

    # 7. 训练循环
    print(f"Starting training with {args.epochs} epochs (from epoch {start_epoch + 1})...")
    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        deam_total = 0
        mtg_total = 0
        deam_ccc = 0.0

        # Step-window accumulators (reset after each optimizer.step)
        step_loss_sum = 0.0
        step_samples = 0
        step_deam_loss_sum = 0.0
        step_deam_samples = 0
        step_mtg_loss_sum = 0.0
        step_mtg_samples = 0
        step_deam_ccc_sum = 0.0

        # 交替训练两个数据集
        deam_iter = iter(deam_train_loader)
        mtg_iter = iter(mtg_train_loader)
        num_batches = max(len(deam_train_loader), len(mtg_train_loader))

        # Initialize gradient accumulation step counter
        accumulation_counter = 0

        lr = float(optimizer.param_groups[0].get("lr", args.lr))
        train_pbar = tqdm(
            range(num_batches),
            desc=f"Epoch {epoch+1}/{args.epochs} [train]",
            total=num_batches,
            dynamic_ncols=True,
            disable=bool(args.no_progress),
        )

        for batch_idx in train_pbar:
            # 训练DEAM batch
            try:
                deam_features, deam_labels = next(deam_iter)

                # Move data to device
                for key in deam_features:
                    deam_features[key] = deam_features[key].to(device)
                deam_labels = deam_labels.to(device)

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(deam_features)

                # Compute loss and metric
                loss, ccc = loss_fn(outputs, deam_labels, task_type='deam')

                # Task weight
                loss = loss * args.deam_weight

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                loss.backward()

                # Update statistics
                running_loss += (loss.item() * args.accumulation_steps) * deam_features['mel'].size(0)
                deam_total += deam_features['mel'].size(0)
                if ccc is not None:
                    deam_ccc += ccc.item() * deam_features['mel'].size(0)

                # Step-window statistics (use un-normalized loss value)
                deam_bs = int(deam_features['mel'].size(0))
                deam_loss_val = float(loss.item() * args.accumulation_steps)
                step_loss_sum += deam_loss_val * deam_bs
                step_samples += deam_bs
                step_deam_loss_sum += deam_loss_val * deam_bs
                step_deam_samples += deam_bs
                if ccc is not None:
                    step_deam_ccc_sum += float(ccc.item()) * deam_bs

                # Update accumulation counter
                accumulation_counter += 1

            except StopIteration:
                # Reinitialize DEAM iterator if exhausted
                deam_iter = iter(deam_train_loader)
                continue

            # 训练MTG batch
            try:
                mtg_features, mtg_labels = next(mtg_iter)

                # Move data to device
                for key in mtg_features:
                    mtg_features[key] = mtg_features[key].to(device)
                mtg_labels = mtg_labels.to(device)

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(mtg_features)

                # Compute loss
                loss, _ = loss_fn(outputs, mtg_labels, task_type='mtg')

                # Task weight
                loss = loss * args.mtg_weight
                

                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps

                loss.backward()

                # Update statistics
                running_loss += (loss.item() * args.accumulation_steps) * mtg_features['mel'].size(0)
                mtg_total += mtg_features['mel'].size(0)

                # Step-window statistics (use un-normalized loss value)
                mtg_bs = int(mtg_features['mel'].size(0))
                mtg_loss_val = float(loss.item() * args.accumulation_steps)
                step_loss_sum += mtg_loss_val * mtg_bs
                step_samples += mtg_bs
                step_mtg_loss_sum += mtg_loss_val * mtg_bs
                step_mtg_samples += mtg_bs

                # Update accumulation counter
                accumulation_counter += 1

            except StopIteration:
                # Reinitialize MTG iterator if exhausted
                mtg_iter = iter(mtg_train_loader)
                continue

            # Perform optimizer step after accumulation_steps batches
            if accumulation_counter >= args.accumulation_steps:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0

                # Update optimizer step counter and write step-based TensorBoard logs
                optimizer_step += 1
                lr = float(optimizer.param_groups[0].get("lr", args.lr))

                if args.log_interval > 0 and (optimizer_step % int(args.log_interval) == 0):
                    if step_samples > 0:
                        writer.add_scalar('Train/loss_total', float(step_loss_sum) / float(step_samples), optimizer_step)
                    if step_deam_samples > 0:
                        writer.add_scalar('Train/loss_deam', float(step_deam_loss_sum) / float(step_deam_samples), optimizer_step)
                        writer.add_scalar('Train/deam_ccc', float(step_deam_ccc_sum) / float(step_deam_samples), optimizer_step)
                    if step_mtg_samples > 0:
                        writer.add_scalar('Train/loss_mtg', float(step_mtg_loss_sum) / float(step_mtg_samples), optimizer_step)
                    writer.add_scalar('Train/lr', lr, optimizer_step)

                # Reset step-window accumulators
                step_loss_sum = 0.0
                step_samples = 0
                step_deam_loss_sum = 0.0
                step_deam_samples = 0
                step_mtg_loss_sum = 0.0
                step_mtg_samples = 0
                step_deam_ccc_sum = 0.0

            # Update progress bar postfix (cheap, numeric only)
            seen = deam_total + mtg_total
            if seen > 0:
                cur_loss = running_loss / seen
                cur_ccc = (deam_ccc / deam_total) if deam_total > 0 else 0.0
                train_pbar.set_postfix({"loss": f"{cur_loss:.4f}", "deam_ccc": f"{cur_ccc:.3f}", "lr": f"{lr:.1e}"})

        # Make sure to take the last step if accumulation counter is not zero
        if accumulation_counter > 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            optimizer_step += 1
            lr = float(optimizer.param_groups[0].get("lr", args.lr))

            if args.log_interval > 0 and (optimizer_step % int(args.log_interval) == 0):
                if step_samples > 0:
                    writer.add_scalar('Train/loss_total', float(step_loss_sum) / float(step_samples), optimizer_step)
                if step_deam_samples > 0:
                    writer.add_scalar('Train/loss_deam', float(step_deam_loss_sum) / float(step_deam_samples), optimizer_step)
                    writer.add_scalar('Train/deam_ccc', float(step_deam_ccc_sum) / float(step_deam_samples), optimizer_step)
                if step_mtg_samples > 0:
                    writer.add_scalar('Train/loss_mtg', float(step_mtg_loss_sum) / float(step_mtg_samples), optimizer_step)
                writer.add_scalar('Train/lr', lr, optimizer_step)

            step_loss_sum = 0.0
            step_samples = 0
            step_deam_loss_sum = 0.0
            step_deam_samples = 0
            step_mtg_loss_sum = 0.0
            step_mtg_samples = 0
            step_deam_ccc_sum = 0.0

        # Calculate epoch statistics
        epoch_loss = running_loss / (deam_total + mtg_total) if (deam_total + mtg_total) > 0 else 0.0
        avg_deam_ccc = deam_ccc / deam_total if deam_total > 0 else 0.0

        # Keep epoch-level summaries (optional, still useful)
        writer.add_scalar('Training/Loss_epoch', epoch_loss, epoch)
        writer.add_scalar('Training/DEAM_CCC_epoch', avg_deam_ccc, epoch)

        # 8. 验证循环
        model.eval()
        val_t0 = time.perf_counter()

        val_loss = 0.0
        val_deam_total = 0
        val_mtg_total = 0
        val_deam_ccc = 0.0

        deam_val_preds = []
        deam_val_targets = []
        mtg_val_scores = []
        mtg_val_targets = []

        with torch.no_grad():
            # Validate DEAM
            deam_val_pbar = tqdm(
                deam_val_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [val:DEAM]",
                total=len(deam_val_loader),
                dynamic_ncols=True,
                leave=False,
                disable=bool(args.no_progress),
            )
            for features, labels in deam_val_pbar:
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(features)
                loss, ccc = loss_fn(outputs, labels, task_type='deam')

                loss = loss * args.deam_weight

                val_loss += loss.item() * features['mel'].size(0)
                val_deam_total += features['mel'].size(0)
                if ccc is not None:
                    val_deam_ccc += ccc.item() * features['mel'].size(0)

                deam_val_preds.append(outputs['regression'].detach().cpu().float().numpy())
                deam_val_targets.append(labels.detach().cpu().float().numpy())

                if val_deam_total > 0:
                    deam_val_pbar.set_postfix({"loss": f"{(val_loss / max(1, val_deam_total + val_mtg_total)):.4f}", "ccc": f"{(val_deam_ccc / val_deam_total):.3f}"})

            # Validate MTG
            mtg_val_pbar = tqdm(
                mtg_val_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [val:MTG]",
                total=len(mtg_val_loader),
                dynamic_ncols=True,
                leave=False,
                disable=bool(args.no_progress),
            )
            for features, labels in mtg_val_pbar:
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)

                # Forward pass with automatic mixed precision
                with autocast_ctx:
                    outputs = model(features)
                loss, _ = loss_fn(outputs, labels, task_type='mtg')

                loss = loss * args.mtg_weight

                val_loss += loss.item() * features['mel'].size(0)
                val_mtg_total += features['mel'].size(0)

                mtg_logits = outputs['classification'].detach().cpu().float().numpy()
                mtg_probs = 1.0 / (1.0 + np.exp(-mtg_logits))
                mtg_val_scores.append(mtg_probs)
                mtg_val_targets.append(labels.detach().cpu().numpy())

                if val_mtg_total > 0:
                    mtg_val_pbar.set_postfix({"loss": f"{(val_loss / max(1, val_deam_total + val_mtg_total)):.4f}"})

        # Calculate validation statistics
        val_epoch_loss = val_loss / (val_deam_total + val_mtg_total) if (val_deam_total + val_mtg_total) > 0 else 0.0
        val_avg_deam_ccc = val_deam_ccc / val_deam_total if val_deam_total > 0 else 0.0

        # Log validation to TensorBoard aligned to optimizer_step axis
        writer.add_scalar('Validation/Loss', val_epoch_loss, optimizer_step)
        writer.add_scalar('Validation/DEAM_CCC', val_avg_deam_ccc, optimizer_step)

        # Compute epoch-level DEAM metrics on full validation set
        if deam_val_preds and deam_val_targets:
            deam_pred = np.concatenate(deam_val_preds, axis=0)
            deam_tgt = np.concatenate(deam_val_targets, axis=0)
            writer.add_scalar('Validation/DEAM_RMSE', _rmse_np(deam_pred, deam_tgt), optimizer_step)
            writer.add_scalar('Validation/DEAM_Pearson', _pearson_np(deam_pred, deam_tgt), optimizer_step)
            writer.add_scalar('Validation/DEAM_CCC_epoch', _ccc_np(deam_pred, deam_tgt), optimizer_step)

        # Compute epoch-level MTG metrics on full validation set
        if mtg_val_scores and mtg_val_targets:
            y_score = np.concatenate(mtg_val_scores, axis=0)
            y_true = np.concatenate(mtg_val_targets, axis=0).astype(np.int32)

            mtg_threshold, _ = _search_global_threshold_f1_micro(y_true=y_true, y_score=y_score)
            mtg_m = _mtg_metrics(y_true=y_true, y_score=y_score, threshold=mtg_threshold)

            # Minimal MTG debug print (use only a small prefix to keep it cheap/readable)
            if args.mtg_debug:
                dbg_n = int(min(16, y_true.shape[0]))
                if dbg_n > 0:
                    dbg_score = y_score[:dbg_n]
                    dbg_true = y_true[:dbg_n]
                    dbg_pred = _binarize_with_top1_fallback(dbg_score, mtg_threshold)
                    print(
                        "MTG debug "
                        f"(n={dbg_n}, thr={mtg_threshold:.2f}, top1_fallback=on): "
                        f"prob_mean={dbg_score.mean():.4f} prob_max={dbg_score.max():.4f} | "
                        f"pred_pos_density={dbg_pred.mean():.6f} true_pos_density={dbg_true.mean():.6f} | "
                        f"pred_pos/sample={dbg_pred.sum(axis=1).mean():.3f} true_pos/sample={dbg_true.sum(axis=1).mean():.3f}"
                    )

            writer.add_scalar('Validation/MTG_Threshold', mtg_threshold, optimizer_step)

            writer.add_scalar('Validation/MTG_ROC_AUC_micro', mtg_m['roc_auc_micro'], optimizer_step)
            writer.add_scalar('Validation/MTG_ROC_AUC_macro', mtg_m['roc_auc_macro'], optimizer_step)
            writer.add_scalar('Validation/MTG_PR_AUC_micro', mtg_m['pr_auc_micro'], optimizer_step)
            writer.add_scalar('Validation/MTG_PR_AUC_macro', mtg_m['pr_auc_macro'], optimizer_step)
            writer.add_scalar('Validation/MTG_F1_micro', mtg_m['f1_micro'], optimizer_step)
            writer.add_scalar('Validation/MTG_F1_macro', mtg_m['f1_macro'], optimizer_step)
            writer.add_scalar('Validation/MTG_Precision_micro', mtg_m['precision_micro'], optimizer_step)
            writer.add_scalar('Validation/MTG_Precision_macro', mtg_m['precision_macro'], optimizer_step)
            writer.add_scalar('Validation/MTG_Recall_micro', mtg_m['recall_micro'], optimizer_step)
            writer.add_scalar('Validation/MTG_Recall_macro', mtg_m['recall_macro'], optimizer_step)

            # Track best threshold across epochs (by val F1-micro)
            if mtg_m["f1_micro"] > best_mtg_val_f1_micro:
                best_mtg_val_f1_micro = float(mtg_m["f1_micro"])
                best_mtg_threshold = float(mtg_threshold)
                best_mtg_epoch = int(epoch)

        # Print concise epoch summary (important metrics only)
        val_dt = time.perf_counter() - val_t0
        epoch_dt = time.perf_counter() - epoch_t0

        deam_rmse = _rmse_np(deam_pred, deam_tgt) if (deam_val_preds and deam_val_targets) else float("nan")
        deam_pearson = _pearson_np(deam_pred, deam_tgt) if (deam_val_preds and deam_val_targets) else float("nan")
        deam_ccc_epoch = _ccc_np(deam_pred, deam_tgt) if (deam_val_preds and deam_val_targets) else float("nan")

        mtg_line = ""
        if mtg_val_scores and mtg_val_targets:
            mtg_line = (
                f" | MTG(thr={mtg_threshold:.2f}) "
                f"F1μ={mtg_m['f1_micro']:.4f} PR-AUCμ={mtg_m['pr_auc_micro']:.4f} ROC-AUCμ={mtg_m['roc_auc_micro']:.4f}"
            )

        best_line = ""
        if best_mtg_threshold is not None and best_mtg_epoch is not None:
            best_line = (
                f" | best(MTG F1μ={best_mtg_val_f1_micro:.4f} thr={best_mtg_threshold:.2f} @ep={best_mtg_epoch+1})"
            )

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train: loss={epoch_loss:.4f} deam_ccc={avg_deam_ccc:.4f} "
            f"| val: loss={val_epoch_loss:.4f} deam_ccc={val_avg_deam_ccc:.4f} "
            f"rmse={deam_rmse:.4f} r={deam_pearson:.4f} ccc={deam_ccc_epoch:.4f}"
            f"{mtg_line}{best_line}"
        )
        print(f"time: epoch={epoch_dt:.1f}s (val={val_dt:.1f}s) lr={lr:.1e}")

        # 8.1 Save checkpoint at end of each epoch
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint_state = {
            'epoch': epoch,
            'optimizer_step': optimizer_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_mtg_val_f1_micro': best_mtg_val_f1_micro,
            'best_mtg_threshold': best_mtg_threshold,
            'best_mtg_epoch': best_mtg_epoch,
            'fusion_type': args.fusion_type,
            'num_mtg_tags': num_mtg_tags,
            'mood_tags': getattr(mtg_train_loader.dataset, "mood_tags", None),
            'args': vars(args),
        }

        # Always save last checkpoint (for resuming interrupted training)
        last_ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pt")
        torch.save(checkpoint_state, last_ckpt_path)

        # Save best checkpoint (based on validation loss)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            checkpoint_state['best_val_loss'] = best_val_loss
            best_ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
            torch.save(checkpoint_state, best_ckpt_path)
            print(f"New best model saved (val_loss={best_val_loss:.4f}) -> {best_ckpt_path}")

        print("-" * 60)

    # 9. 测试阶段
    print("\nStarting final testing...")
    model.eval()
    test_loss = 0.0
    test_deam_total = 0
    test_mtg_total = 0
    test_deam_ccc = 0.0

    deam_test_preds = []
    deam_test_targets = []
    mtg_test_scores = []
    mtg_test_targets = []

    with torch.no_grad():
        # Test DEAM
        for features, labels in deam_test_loader:
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            with autocast_ctx:
                outputs = model(features)
            loss, ccc = loss_fn(outputs, labels, task_type='deam')

            loss = loss * args.deam_weight

            test_loss += loss.item() * features['mel'].size(0)
            test_deam_total += features['mel'].size(0)
            if ccc is not None:
                test_deam_ccc += ccc.item() * features['mel'].size(0)

            deam_test_preds.append(outputs['regression'].detach().cpu().float().numpy())
            deam_test_targets.append(labels.detach().cpu().float().numpy())

        # Test MTG
        for features, labels in mtg_test_loader:
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            with autocast_ctx:
                outputs = model(features)
            loss, _ = loss_fn(outputs, labels, task_type='mtg')

            loss = loss * args.mtg_weight

            test_loss += loss.item() * features['mel'].size(0)
            test_mtg_total += features['mel'].size(0)

            mtg_logits = outputs['classification'].detach().cpu().float().numpy()
            mtg_probs = 1.0 / (1.0 + np.exp(-mtg_logits))
            mtg_test_scores.append(mtg_probs)
            mtg_test_targets.append(labels.detach().cpu().numpy())

    # Calculate test statistics
    test_epoch_loss = test_loss / (test_deam_total + test_mtg_total) if (test_deam_total + test_mtg_total) > 0 else 0.0
    test_avg_deam_ccc = test_deam_ccc / test_deam_total if test_deam_total > 0 else 0.0

    # Log test to TensorBoard
    writer.add_scalar('Test/Loss', test_epoch_loss, optimizer_step)
    writer.add_scalar('Test/DEAM_CCC', test_avg_deam_ccc, optimizer_step)

    if deam_test_preds and deam_test_targets:
        deam_pred = np.concatenate(deam_test_preds, axis=0)
        deam_tgt = np.concatenate(deam_test_targets, axis=0)
        writer.add_scalar('Test/DEAM_RMSE', _rmse_np(deam_pred, deam_tgt), optimizer_step)
        writer.add_scalar('Test/DEAM_Pearson', _pearson_np(deam_pred, deam_tgt), optimizer_step)
        writer.add_scalar('Test/DEAM_CCC_epoch', _ccc_np(deam_pred, deam_tgt), optimizer_step)

    if mtg_test_scores and mtg_test_targets:
        y_score = np.concatenate(mtg_test_scores, axis=0)
        y_true = np.concatenate(mtg_test_targets, axis=0).astype(np.int32)
        test_threshold = float(best_mtg_threshold) if best_mtg_threshold is not None else 0.5
        mtg_m = _mtg_metrics(y_true=y_true, y_score=y_score, threshold=test_threshold)
        writer.add_scalar('Test/MTG_Threshold', test_threshold, optimizer_step)
        writer.add_scalar('Test/MTG_ROC_AUC_micro', mtg_m['roc_auc_micro'], optimizer_step)
        writer.add_scalar('Test/MTG_ROC_AUC_macro', mtg_m['roc_auc_macro'], optimizer_step)
        writer.add_scalar('Test/MTG_PR_AUC_micro', mtg_m['pr_auc_micro'], optimizer_step)
        writer.add_scalar('Test/MTG_PR_AUC_macro', mtg_m['pr_auc_macro'], optimizer_step)
        writer.add_scalar('Test/MTG_F1_micro', mtg_m['f1_micro'], optimizer_step)
        writer.add_scalar('Test/MTG_F1_macro', mtg_m['f1_macro'], optimizer_step)
        writer.add_scalar('Test/MTG_Precision_micro', mtg_m['precision_micro'], optimizer_step)
        writer.add_scalar('Test/MTG_Precision_macro', mtg_m['precision_macro'], optimizer_step)
        writer.add_scalar('Test/MTG_Recall_micro', mtg_m['recall_micro'], optimizer_step)
        writer.add_scalar('Test/MTG_Recall_macro', mtg_m['recall_macro'], optimizer_step)

    # Print test summary
    print(f"Test Loss: {test_epoch_loss:.4f}")
    print(f"Test DEAM CCC: {test_avg_deam_ccc:.4f}")
    if deam_test_preds and deam_test_targets:
        print(f"Test DEAM RMSE: {_rmse_np(deam_pred, deam_tgt):.4f}")
        print(f"Test DEAM Pearson: {_pearson_np(deam_pred, deam_tgt):.4f}")
        print(f"Test DEAM CCC (epoch): {_ccc_np(deam_pred, deam_tgt):.4f}")
    if mtg_test_scores and mtg_test_targets:
        print(
            "Test MTG "
            f"thr={test_threshold:.2f} (top1_fallback=on) | "
            f"ROC-AUC micro/macro: {mtg_m['roc_auc_micro']:.4f}/{mtg_m['roc_auc_macro']:.4f} | "
            f"PR-AUC micro/macro: {mtg_m['pr_auc_micro']:.4f}/{mtg_m['pr_auc_macro']:.4f} | "
            f"F1 micro/macro: {mtg_m['f1_micro']:.4f}/{mtg_m['f1_macro']:.4f} | "
            f"P micro/macro: {mtg_m['precision_micro']:.4f}/{mtg_m['precision_macro']:.4f} | "
            f"R micro/macro: {mtg_m['recall_micro']:.4f}/{mtg_m['recall_macro']:.4f}"
        )
    print("-" * 60)

    # 10. 模型保存与清理
    model_save_path = f"unified_mood_model_{args.fusion_type}.pt"
    mood_tags = getattr(mtg_train_loader.dataset, "mood_tags", None)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "fusion_type": args.fusion_type,
        "num_mtg_tags": num_mtg_tags,
        "mood_tags": mood_tags,
        "best_mtg_threshold": best_mtg_threshold,
        "best_mtg_val_f1_micro": best_mtg_val_f1_micro,
        "best_mtg_epoch": best_mtg_epoch,
    }
    torch.save(ckpt, model_save_path)
    writer.close()

    print(f"\nTraining completed!")
    print(f"Model saved to: {model_save_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == "__main__":
    main()
