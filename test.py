import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from tqdm import tqdm

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--path",  required=True, type=str, help="Project root path")
parser.add_argument("--split", required=True, type=str, help="Data split: train / val / test")
args = parser.parse_args()

PATH     = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT    = args.split

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
IMAGE_SIZE = 320
NICKNAME   = "Allison"
THRESHOLD  = 0.3
THRESHOLDS = [0.40, 0.55, 0.50, 0.65, 0.75]  # per-class, tuned on validation set

mlb    = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ── Dataset ───────────────────────────────────────────────────────────────────
class Dataset(data.Dataset):
    def __init__(self, list_IDs, target_type):
        self.list_IDs    = list_IDs
        self.target_type = target_type
        self.normalize   = transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y  = xdf_dset_test.target_class.get(ID)

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y.split(",")]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)
            labels_ohe[y] = 1
        y = torch.FloatTensor(labels_ohe)

        file = DATA_DIR + xdf_dset_test.id.get(ID)
        img  = cv2.imread(file)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img  = img / 255.0
        X    = torch.FloatTensor(img).permute(2, 0, 1)
        return self.normalize(X), y


# ── Data loader ───────────────────────────────────────────────────────────────
def read_data(target_type):
    test_ids = list(xdf_dset_test.index)
    loader   = data.DataLoader(
        Dataset(test_ids, target_type),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return loader


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
    model.load_state_dict(torch.load(f"model_{NICKNAME}.pt", map_location=device))
    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    return model, criterion


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
        val    = fn_map[m](y_true, y_pred) if m in fn_map else 0
        res[m] = val
        total += val
    if 'sum' in aggregates:
        res['sum'] = total
    if 'avg' in aggregates:
        res['avg'] = total / len(metrics)
    return res


# ── Inference ─────────────────────────────────────────────────────────────────
def test_model(test_ds, list_of_metrics, list_of_agg):
    model, criterion = model_definition()
    test_loss, steps = 0, 0
    all_logits, all_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_ds), desc="Inference") as pbar:
            for xdata, xtarget in test_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)
                output     = model(xdata)
                loss       = criterion(output, xtarget)
                test_loss += loss.item()
                steps     += 1
                all_logits = np.vstack((all_logits, output.detach().cpu().numpy()))
                all_labels = np.vstack((all_labels, xtarget.cpu().numpy()))
                pbar.update(1)
                pbar.set_postfix_str(f"loss={test_loss/steps:.5f}")

    pred_probs  = 1 / (1 + np.exp(-all_logits[1:]))
    pred_labels = np.zeros_like(pred_probs)
    for c in range(OUTPUTS_a):
        pred_labels[:, c] = (pred_probs[:, c] >= THRESHOLDS[c]).astype(int)

    test_metrics = metrics_func(list_of_metrics, list_of_agg, all_labels[1:], pred_labels)
    for met, val in test_metrics.items():
        print(f"Test {met}: {val:.5f}")

    results = xdf_dset_test.copy()
    results['results'] = [
        ",".join(str(int(e)) for e in row) for row in pred_labels
    ]
    results.to_excel(f"results_{NICKNAME}.xlsx", index=False)
    print(f"Results saved to results_{NICKNAME}.xlsx")


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
    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()
    OUTPUTS_a     = len(class_names)

    test_ds = read_data(target_type=2)
    test_model(test_ds, list_of_metrics=['f1_macro'], list_of_agg=['avg'])
