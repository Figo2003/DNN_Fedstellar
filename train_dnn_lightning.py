"""
train_dnn_lightning.py

Requirements (install if needed):
pip install torch torchvision torchaudio pytorch-lightning scikit-learn pandas numpy torchmetrics

Usage examples:
python train_dnn_lightning.py --data_path /path/to/cic_csv_folder_or_file --batch_size 512 --max_epochs 30
"""

import os
import glob
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

class CICIDS2017DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        target_col: str = "Label",
        numeric_cols: Optional[List[str]] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 512,
        n_jobs: int = 0,
        use_ipca: bool = False,
        ipca_n_components: int = 32,
        ipca_batch_size: int = 1000,
        random_state: int = 42,
    ):
        """
        data_path: path to a single CSV file or a directory containing CSVs (CIC-IDS2017 exports).
        target_col: name of the label column in the CSV(s).
        numeric_cols: list of numeric columns to use. If None, infer numeric columns automatically.
        use_ipca: if True, apply IncrementalPCA to numeric features (useful for large datasets).
        """
        super().__init__()
        self.data_path = data_path
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.use_ipca = use_ipca
        self.ipca_n_components = ipca_n_components
        self.ipca_batch_size = ipca_batch_size
        self.random_state = random_state

        self.scaler = None
        self.label_encoder = None
        self.input_dim = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_csvs(self, path):
        if os.path.isdir(path):
            csvs = sorted(glob.glob(os.path.join(path, "*.csv")))
            if len(csvs) == 0:
                raise FileNotFoundError(f"No CSV files found in directory {path}")
            dfs = [pd.read_csv(p) for p in csvs]
            df = pd.concat(dfs, ignore_index=True)
        elif os.path.isfile(path):
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError(path)
        return df

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        df = self._load_csvs(self.data_path)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the data columns.")

        if self.numeric_cols:
            X_df = df[self.numeric_cols].copy()
        else:
            numeric_mask = df.select_dtypes(include=[np.number]).columns.tolist()

            if self.target_col in numeric_mask:
                numeric_mask.remove(self.target_col)
            X_df = df[numeric_mask].copy()

        df = df.loc[X_df.dropna().index].reset_index(drop=True)
        X_df = X_df.loc[df.index].reset_index(drop=True)  # sync
        y_series = df[self.target_col].reset_index(drop=True)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_series)

        X = X_df.values.astype(np.float32)

        #X = X_df.values.astype(np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)



        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.use_ipca:
            ipca = IncrementalPCA(n_components=self.ipca_n_components)
            n_samples = X_scaled.shape[0]
            bs = self.ipca_batch_size
            for start in range(0, n_samples, bs):
                end = min(start + bs, n_samples)
                ipca.partial_fit(X_scaled[start:end])
            X_reduced = np.empty((n_samples, self.ipca_n_components), dtype=np.float32)
            for start in range(0, n_samples, bs):
                end = min(start + bs, n_samples)
                X_reduced[start:end] = ipca.transform(X_scaled[start:end])
            X_final = X_reduced
        else:
            X_final = X_scaled

        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_final, y, test_size=(self.test_size + self.val_size), random_state=self.random_state, stratify=stratify
        )

        val_rel = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_rel), random_state=self.random_state, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )

        self.input_dim = X_train.shape[1]

        self.train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        self.test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        print(f"[DataModule] input_dim={self.input_dim}, train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
        print(f"[DataModule] Label classes: {list(self.label_encoder.classes_)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs)

class DNNLightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=True, on_epoch=True)
        return {"test_loss": loss}

def main(args):
    pl.seed_everything(43)

    dm = CICIDS2017DataModule(
        data_path=args.data_path,
        target_col=args.target_col,
        numeric_cols=None,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        n_jobs=max(0, args.num_workers),
        use_ipca=args.use_ipca,
        ipca_n_components=args.ipca_n_components,
        ipca_batch_size=args.ipca_batch_size,
        random_state=42,
    )
    dm.setup()

    num_classes = len(dm.label_encoder.classes_)
    model = DNNLightning(input_dim=dm.input_dim, num_classes=num_classes, hidden_dims=list(map(int, args.hidden_dims.split(","))), dropout=args.dropout, lr=args.lr)
    
    logger = TensorBoardLogger("tb_logs", name="cic_dnn")
    print([1,2,3,4])
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        logger=logger,
        log_every_n_steps=10,
        precision=16 if args.use_fp16 and torch.cuda.is_available() else 32,
    )
    print([1,2,3,4])
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    print("=== Test Results ===")
    print(test_results)
    print([1,2,3,4])
    plotting("/Users/yousseffarag/Documents/Dodis/Bachelorarbeit/Databases/tb_logs")
    
def plotting(log_dir):
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events")]
    ea = event_accumulator.EventAccumulator(os.path.join(log_dir, event_files[0]))
    ea.Reload()

    train_loss = [e.value for e in ea.Scalars("train_loss_epoch")]
    val_loss   = [e.value for e in ea.Scalars("val_loss")]
    epochs     = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="CSV file or directory containing CIC-IDS2017 CSV files")
    parser.add_argument("--target_col", type=str, default="Label", help="Name of label column")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="256,128", help="Comma-separated hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--use_ipca", action="store_true", help="Use Incremental PCA for dimensionality reduction")
    parser.add_argument("--ipca_n_components", type=int, default=32)
    parser.add_argument("--ipca_batch_size", type=int, default=1000)
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision (fp16) if available")
    args = parser.parse_args()
    main(args)
    



