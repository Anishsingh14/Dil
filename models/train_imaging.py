import csv
import io
import os
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from azure.core.exceptions import AzureError
from azure.storage.blob import ContainerClient
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


CHEXPERT_SAS_URL_ENV = "CHEXPERT_SAS_URL"
LABELS_BLOB_NAME = "labels.csv"
IMAGE_DIR_PREFIX = "images/"


class CheXpertPlusBlobClient:
    """Lightweight wrapper around Azure Blob ContainerClient for CheXpert Plus."""

    def __init__(self):
        self._sas_url = os.environ.get(CHEXPERT_SAS_URL_ENV)
        if not self._sas_url:
            raise RuntimeError(
                f"Missing required environment variable: {CHEXPERT_SAS_URL_ENV}. "
                "Set it to the SAS URL for the CheXpert Plus Azure Blob container."
            )
        self._container_client: Optional[ContainerClient] = None
        self._blob_names: Optional[List[str]] = None
        self._labels: Optional[Dict[str, float]] = None

    @property
    def container_client(self) -> ContainerClient:
        if self._container_client is None:
            self._container_client = ContainerClient.from_container_url(self._sas_url)
        return self._container_client

    def list_image_blobs(self) -> List[str]:
        """List and cache all image blob names under the image directory."""
        if self._blob_names is not None:
            return self._blob_names

        blob_names = []
        try:
            for blob in self.container_client.list_blobs(name_starts_with=IMAGE_DIR_PREFIX):
                if blob.name.lower().endswith((".png", ".jpg", ".jpeg", ".dicom", ".dcm")):
                    blob_names.append(blob.name)
        except AzureError as e:
            raise RuntimeError("Failed to list blobs from CheXpert Plus container") from e

        if not blob_names:
            raise RuntimeError(f"No image blobs found under prefix '{IMAGE_DIR_PREFIX}'")

        self._blob_names = blob_names
        return blob_names

    def load_labels(self) -> Dict[str, float]:
        """Load and cache labels from labels.csv blob into memory."""
        if self._labels is not None:
            return self._labels

        try:
            blob_client = self.container_client.get_blob_client(LABELS_BLOB_NAME)
            stream = blob_client.download_blob()
            content = stream.readall().decode("utf-8")
        except AzureError as e:
            raise RuntimeError(f"Failed to download labels blob '{LABELS_BLOB_NAME}'") from e

        reader = csv.DictReader(io.StringIO(content))
        labels = {}
        for row in reader:
            image_path = row.get("Path") or row.get("Image") or row.get("image_path")
            label_str = row.get("Label") or row.get("Target") or row.get("label") or row.get("target")
            if image_path and label_str is not None:
                try:
                    labels[image_path.strip()] = float(label_str)
                except ValueError:
                    continue

        if not labels:
            raise RuntimeError("No valid labels parsed from labels.csv")

        self._labels = labels
        return labels

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def download_blob_bytes(self, blob_name: str) -> bytes:
        """Download a single blob's bytes with retry logic."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            stream = blob_client.download_blob()
            return stream.readall()
        except AzureError as e:
            raise RuntimeError("Blob download failed") from e


_chexpert_client: Optional[CheXpertPlusBlobClient] = None


def get_chexpert_client() -> CheXpertPlusBlobClient:
    global _chexpert_client
    if _chexpert_client is None:
        _chexpert_client = CheXpertPlusBlobClient()
    return _chexpert_client


class StreamedCheXpertPlusDataset(IterableDataset):
    """IterableDataset streaming CheXpert Plus images from Azure Blob Storage."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 224,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.seed = seed

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._client = get_chexpert_client()
        self._all_blob_names = self._client.list_image_blobs()
        self._labels = self._client.load_labels()

        self._split_indices = self._compute_split_indices()

    def _compute_split_indices(self) -> Tuple[int, int]:
        n = len(self._all_blob_names)
        train_end = int(n * self.train_ratio)
        if self.split == "train":
            return 0, train_end
        elif self.split in ("val", "validation"):
            return train_end, n
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        start, end = self._split_indices
        count = 0
        max_count = self.max_samples or (end - start)

        np.random.seed(self.seed)
        indices = np.random.permutation(range(start, end))

        for idx in indices:
            if count >= max_count:
                return

            blob_name = self._all_blob_names[idx]
            label = self._labels.get(blob_name)
            if label is None:
                continue

            try:
                image_bytes = self._client.download_blob_bytes(blob_name)
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_tensor = self.preprocess(img)
                yield img_tensor, torch.tensor(label, dtype=torch.float32)
                count += 1

            except Exception:
                continue


def create_dataloader(
    dataset: IterableDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_imaging_model(
    model_path: str = "models/imaging_densenet121_lstm.pth",
    metadata_path: str = "models/imaging_metadata.json",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_samples_per_epoch: int = 10000,
    val_samples: int = 2000,
    device: Optional[str] = None,
    sequence_length: int = 1,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    from models.imaging_inference import DenseNet121LSTM

    model = DenseNet121LSTM(
        num_classes=1,
        lstm_hidden=256,
        lstm_layers=1,
        dropout=0.3,
        pretrained=True,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    train_dataset = StreamedCheXpertPlusDataset(
        split="train",
        max_samples=max_samples_per_epoch,
    )
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)

    val_dataset = StreamedCheXpertPlusDataset(
        split="val",
        max_samples=val_samples,
    )
    val_loader = create_dataloader(val_dataset, batch_size=batch_size)

    print(f"Starting training for {epochs} epochs...")
    print(f"Max samples per epoch: {max_samples_per_epoch}")
    print(f"Validation samples: {val_samples}")
    print(f"Total images available: {len(get_chexpert_client().list_image_blobs())}")

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            if batch_idx * batch_size >= max_samples_per_epoch:
                break

            images = images.unsqueeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_idx, (images, labels) in enumerate(pbar):
                if batch_idx * batch_size >= val_samples:
                    break

                images = images.unsqueeze(1).to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_version": "imaging-densenet121-lstm-v1.1.0",
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
            }, model_path)
            print(f"  -> Best model saved to {model_path}")

    metadata = {
        "model_version": "imaging-densenet121-lstm-v1.1.0",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_val_loss": best_val_loss,
        "final_val_acc": val_acc,
        "dataset": "CheXpert Plus (Azure Blob)",
        "sequence_length": sequence_length,
    }

    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. Metadata saved to {metadata_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_imaging_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples_per_epoch=args.max_samples,
        val_samples=args.val_samples,
        device=device,
    )