import random
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(16)
    torch.backends.cudnn.deterministic = True

# ── Paths ─────────────────────────────────────────────────────────────────────
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
os.chdir(OR_PATH)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPOCH    = 20
BATCH_SIZE = 8
LR         = 1e-4
IMAGE_SIZE = 320
THRESHOLD  = 0.3
NICKNAME   = "Allison"
SAVE_MODEL = True

mlb    = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ── Dataset ───────────────────────────────────────────────────────────────────
class Dataset(data.Dataset):
    def __init__(self, list_IDs, type_data, target_type):
        self.type_data   = type_data
        self.list_IDs    = list_IDs
        self.target_type = target_type
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomErasing(p=0.1),
        ])
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID  = self.list_IDs[index]
        src = xdf_dset if self.type_data == 'train' else xdf_dset_test
        y   = src.target_class.get(ID)

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y.split(",")]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)
            labels_ohe[y] = 1
        y = torch.FloatTensor(labels_ohe)

        file = DATA_DIR + src.id.get(ID)
        img  = cv2.imread(file)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        X   = torch.FloatTensor(img).permute(2, 0, 1)

        if self.type_data == 'train':
            X = self.train_transforms(X)
        return self.normalize(X), y


# ── Data loaders ──────────────────────────────────────────────────────────────
def read_data(target_type):
    train_ids = list(xdf_dset.index)
    test_ids  = list(xdf_dset_test.index)

    train_params = {'batch_size': BATCH_SIZE, 'shuffle': True}
    test_params  = {'batch_size': BATCH_SIZE, 'shuffle': False}

    train_loader = data.DataLoader(Dataset(train_ids, 'train', target_type), **train_params)
    test_loader  = data.DataLoader(Dataset(test_ids,  'test',  target_type), **test_params)
    return train_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────────
def model_definition():
    model = models.densenet169(weights='DenseNet169_Weights.DEFAULT')
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, OUTPUTS_a)
    )
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    return model, optimizer, criterion, scheduler


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics_func(metrics, aggregates, y_true, y_pred):
    fn_map = {
        'f1_micro':    lambda a, b: f1_score(a, b, average='micro'),
        'f1_macro':    lambda a, b: f1_score(a, b, average='macro'),
        'f1_weighted': lambda a, b: f1_score(a, b, average='weighted'),
        'coh':         cohen_kappa_score,
        'acc':         accuracy_score,
        'mat':         matthews_corrcoef,
        'hlm':         hamming_loss,
    }
    res, total = {}, 0
    for m in metrics:
        val      = fn_map[m](y_true, y_pred) if m in fn_map else 0
        res[m]   = val
        total   += val
    if 'sum' in aggregates:
        res['sum'] = total
    if 'avg' in aggregates:
        res['avg'] = total / len(metrics)
    return res


# ── Per-class threshold search ────────────────────────────────────────────────
def find_best_thresholds(probs, labels):
    thresholds = []
    for c in range(OUTPUTS_a):
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, c] >= t).astype(int)
            f1    = f1_score(labels[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds


# ── Training loop ─────────────────────────────────────────────────────────────
def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on):
    model, optimizer, criterion, scheduler = model_definition()
    met_test_best = 0
    start_epoch   = 0

    ckpt_path = f"checkpoint_{NICKNAME}.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        met_test_best = ckpt['best_f1']
        print(f"Resumed from epoch {start_epoch}, best f1={met_test_best:.5f}")

    patience, no_improve = 8, 0

    for epoch in range(start_epoch, N_EPOCH):

        # ── Train ──
        model.train()
        train_loss, steps = 0, 0
        all_logits, all_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        with tqdm(total=len(train_ds), desc=f"Epoch {epoch} [train]") as pbar:
            for xdata, xtarget in train_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)
                optimizer.zero_grad()
                output = model(xdata)
                loss   = criterion(output, xtarget)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                steps      += 1
                all_logits  = np.vstack((all_logits, output.detach().cpu().numpy()))
                all_labels  = np.vstack((all_labels, xtarget.cpu().numpy()))
                pbar.update(1)
                pbar.set_postfix_str(f"loss={train_loss/steps:.5f}")

        train_probs  = 1 / (1 + np.exp(-all_logits[1:]))
        train_preds  = (train_probs >= THRESHOLD).astype(int)
        train_metrics = metrics_func(list_of_metrics, list_of_agg, all_labels[1:], train_preds)

        # ── Evaluate ──
        model.eval()
        test_loss, steps = 0, 0
        all_logits, all_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc=f"Epoch {epoch} [eval] ") as pbar:
                for xdata, xtarget in test_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    output    = model(xdata)
                    loss      = criterion(output, xtarget)
                    test_loss += loss.item()
                    steps     += 1
                    all_logits = np.vstack((all_logits, output.detach().cpu().numpy()))
                    all_labels = np.vstack((all_labels, xtarget.cpu().numpy()))
                    pbar.update(1)
                    pbar.set_postfix_str(f"loss={test_loss/steps:.5f}")

        test_probs  = 1 / (1 + np.exp(-all_logits[1:]))
        thresholds  = find_best_thresholds(test_probs, all_labels[1:])
        test_preds  = np.zeros_like(test_probs)
        for c in range(OUTPUTS_a):
            test_preds[:, c] = (test_probs[:, c] >= thresholds[c]).astype(int)

        test_metrics = metrics_func(list_of_metrics, list_of_agg, all_labels[1:], test_preds)
        scheduler.step()

        log = f"Epoch {epoch}:"
        log += "".join(f"  Train {k} {v:.5f}" for k, v in train_metrics.items())
        log += " |"
        log += "".join(f"  Test {k} {v:.5f}"  for k, v in test_metrics.items())
        print(log)
        print(f"  Thresholds: {thresholds}")

        met_test = test_metrics.get(save_on, 0)
        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")
            torch.save({
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch':     epoch,
                'best_f1':   met_test,
            }, ckpt_path)

            results = xdf_dset_test.copy()
            results['results'] = [
                ",".join(str(int(e)) for e in row) for row in test_preds
            ]
            results.to_excel(f"results_{NICKNAME}.xlsx", index=False)
            print("  Model saved.")
            met_test_best = met_test
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


# ── Target processing ─────────────────────────────────────────────────────────
def process_target(target_type):
    if target_type == 2:
        target       = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        class_names  = mlb.classes_
        xdf_data['target_class'] = [
            ",".join(str(e) for e in row) for row in final_target
        ]
    elif target_type == 1:
        le = LabelEncoder()
        xdf_data['target_class'] = le.fit_transform(xdf_data['target'])
        class_names = list(xdf_data['target'].unique())
    return class_names


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    excel_dir = PATH + os.path.sep + "excel"
    FILE_NAME = next(
        PATH + os.path.sep + "excel" + os.path.sep + f
        for f in os.listdir(excel_dir) if f.endswith('.xlsx')
    )

    xdf_data      = pd.read_excel(FILE_NAME)
    class_names   = process_target(target_type=2)
    xdf_dset      = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()
    train_ds, test_ds = read_data(target_type=2)

    OUTPUTS_a = len(class_names)

    targets    = np.array([
        [int(e) for e in row.split(',')]
        for row in xdf_dset['target_class']
    ])
    pos_weight = torch.tensor(
        (targets.shape[0] - targets.sum(0)) / (targets.sum(0) + 1e-6),
        dtype=torch.float32
    ).to(device)

    train_and_test(
        train_ds, test_ds,
        list_of_metrics=['f1_macro'],
        list_of_agg=['avg'],
        save_on='f1_macro'
    )
