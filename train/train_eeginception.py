"""
BNCI2014_001 EEGInception —— 5折交叉验证（0train上CV，best模型分别在1test上评测；汇总均值±标准差）
- 训练与早停：以 val_acc 为标准（同时记录 val_macro_f1）
- 日志：每折 train_log.txt（CSV）
- 模型：best_model_fold{k}.pth
- 测试指标：test_metrics.json
- 汇总：test_summary.json（mean ± std）
- 使用 GroupKFold 按 trial 分组，避免窗口泄漏
"""

import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, RandomSampler
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
)
from braindecode.models import EEGInception
from torch.optim.lr_scheduler import StepLR

# ====================== 基本配置 ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r"E:\wfr\code\EEG\result_eeginception_cv5"
os.makedirs(SAVE_DIR, exist_ok=True)

subjects = list(range(1, 10))
batch_size = 16
lr = 5e-4
epochs = 150
patience = 25
base_seed = 42
step_size = 50
gamma = 0.5
n_splits = 5

def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(base_seed)

# ====================== 数据加载与预处理 ======================
dataset = MOABBDataset("BNCI2014_001", subject_ids=subjects)
print(f"Loaded {len(dataset)} subjects.")

def scale_eeg(data):
    return data * 1e6

transforms = [
    Preprocessor("pick_types", eeg=True),
    Preprocessor(scale_eeg),
    Preprocessor("filter", l_freq=4., h_freq=38.),
    Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000),
]
preprocess(dataset, transforms, n_jobs=-1)

sfreq = dataset.datasets[0].raw.info["sfreq"]
offset = int(-0.5 * sfreq)
windows = create_windows_from_events(dataset, trial_start_offset_samples=offset, preload=True)
split = windows.split("session")
train_full, test_set = split["0train"], split["1test"]
print(f"Train(full)={len(train_full)}, Test={len(test_set)}")

# ====================== Trial 分组提取 ======================
y_all = np.array([y for _, y, _ in train_full])

def extract_groups(ds):
    """从 window_info 提取 trial id，支持 dict / 对象 / list 三种格式。"""
    groups, current_trial = [], -1
    for _, _, wi in ds:
        gid = None
        if isinstance(wi, dict):
            if "i_trial" in wi:
                gid = int(wi["i_trial"])
            elif "trial" in wi:
                gid = int(wi["trial"])
            elif "i_window_in_trial" in wi:
                if wi["i_window_in_trial"] == 0:
                    current_trial += 1
                gid = current_trial
        elif hasattr(wi, "i_trial"):
            gid = int(wi.i_trial)
        elif hasattr(wi, "trial"):
            gid = int(wi.trial)
        elif hasattr(wi, "i_window_in_trial"):
            if wi.i_window_in_trial == 0:
                current_trial += 1
            gid = current_trial
        elif isinstance(wi, (list, np.ndarray)):
            if len(wi) == 3:
                gid = int(wi[-1])
            else:
                current_trial += 1
                gid = current_trial
        if gid is None:
            raise RuntimeError(f"Cannot extract trial id from window_info: {wi}")
        groups.append(gid)
    return np.array(groups, dtype=int)

groups = extract_groups(train_full)
n_groups = len(np.unique(groups))
print(f"Extracted {n_groups} trial groups. Using GroupKFold({n_splits}).")

if n_groups < n_splits:
    raise RuntimeError(f"Insufficient groups for {n_splits}-fold CV ({n_groups} found).")

kfold = GroupKFold(n_splits=n_splits)
split_iter = kfold.split(np.arange(len(train_full)), y_all, groups)

# ====================== DataLoader 构建 ======================
def build_loader(ds, batch_size, shuffle, gen=None):
    """优先使用 generator；兼容旧版 PyTorch。"""
    try:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
            num_workers=0, pin_memory=(device == "cuda")
        )
    except TypeError:
        if shuffle:
            sampler = RandomSampler(ds, generator=gen)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device == "cuda"))
        else:
            return DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))

# ====================== 单折训练 ======================
def train_one_fold(fold_id, train_idx, val_idx):
    fold_seed = base_seed + fold_id
    set_seed(fold_seed)
    g = torch.Generator(); g.manual_seed(fold_seed)

    fold_dir = os.path.join(SAVE_DIR, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    log_path = os.path.join(fold_dir, "train_log.txt")
    best_path = os.path.join(fold_dir, f"best_model_fold{fold_id}.pth")

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)
    train_loader = build_loader(train_ds, batch_size, shuffle=True, gen=g)
    val_loader = build_loader(val_ds, batch_size, shuffle=False)

    n_chans = train_ds[0][0].shape[0]
    n_times = train_ds[0][0].shape[1]
    n_classes = len(np.unique([y for _, y, _ in train_full]))

    model = EEGInception(n_chans=n_chans, n_outputs=n_classes, n_times=n_times).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1\n")

    best_val_acc, best_epoch, no_imp = 0.0, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for X, y, _ in tqdm(train_loader, desc=f"[Fold {fold_id}] Epoch {epoch}/{epochs}", ncols=90):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total
        train_loss = loss_sum / len(train_loader)

        model.eval()
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        v_true, v_pred = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                v_loss_sum += loss.item()
                preds = out.argmax(1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)
                v_true.extend(y.cpu().numpy())
                v_pred.extend(preds.cpu().numpy())

        val_acc = v_correct / v_total
        val_loss = v_loss_sum / len(val_loader)
        val_macro_f1 = f1_score(v_true, v_pred, average="macro")
        scheduler.step()

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_macro_f1:.4f}\n")

        print(f"Fold {fold_id} | Epoch {epoch:03d}: TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f} "
              f"ValF1={val_macro_f1:.3f} TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, no_imp = val_acc, epoch, 0
            torch.save(model.state_dict(), best_path)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Fold {fold_id}: early stop at epoch {epoch} (best_val_acc={best_val_acc:.3f})")
                break

    return {
        "best_model_path": best_path,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "n_chans": int(n_chans),
        "n_times": int(n_times),
        "n_classes": int(n_classes),
    }

# ====================== 测试评估 ======================
def eval_on_test(best_model_path, n_chans, n_times, n_classes, fold_id, fold_dir):
    set_seed(base_seed + fold_id)
    model = EEGInception(n_chans=n_chans, n_outputs=n_classes, n_times=n_times).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Fold {fold_id}: test acc={acc:.3f}, macro_f1={f1:.3f}")
    return metrics

# ====================== 主流程与汇总 ======================
all_test_metrics, fold_summaries = [], []
for k, (tr_idx, va_idx) in enumerate(split_iter):
    print(f"\n===== Fold {k} / {n_splits} =====")
    fold_info = train_one_fold(k, tr_idx, va_idx)
    fold_dir = os.path.join(SAVE_DIR, f"fold_{k}")
    m = eval_on_test(
        fold_info["best_model_path"],
        fold_info["n_chans"], fold_info["n_times"], fold_info["n_classes"],
        fold_id=k, fold_dir=fold_dir
    )
    all_test_metrics.append(m)
    fold_summaries.append(fold_info)

def mean_std(values):
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std

accs = [m["accuracy"] for m in all_test_metrics]
pres = [m["macro_precision"] for m in all_test_metrics]
recs = [m["macro_recall"] for m in all_test_metrics]
f1s = [m["macro_f1"] for m in all_test_metrics]

summary = {
    "test_accuracy_mean": mean_std(accs)[0],
    "test_accuracy_std": mean_std(accs)[1],
    "test_macro_precision_mean": mean_std(pres)[0],
    "test_macro_precision_std": mean_std(pres)[1],
    "test_macro_recall_mean": mean_std(recs)[0],
    "test_macro_recall_std": mean_std(recs)[1],
    "test_macro_f1_mean": mean_std(f1s)[0],
    "test_macro_f1_std": mean_std(f1s)[1],
    "fold_details": fold_summaries,
    "base_seed": base_seed,
}

with open(os.path.join(SAVE_DIR, "test_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n===== Summary (1test) =====")
print(f"Acc      : {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}")
print(f"MacroPrec: {summary['test_macro_precision_mean']:.4f} ± {summary['test_macro_precision_std']:.4f}")
print(f"MacroRec : {summary['test_macro_recall_mean']:.4f} ± {summary['test_macro_recall_std']:.4f}")
print(f"MacroF1  : {summary['test_macro_f1_mean']:.4f} ± {summary['test_macro_f1_std']:.4f}")
print(f"Summary saved to {os.path.join(SAVE_DIR, 'test_summary.json')}")
