import asyncio
import os
import time
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


class StreamedChestXrayDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str = "alkzar90/NIH-Chest-X-ray-dataset",
        split: str = "train",
        batch_size: int = 32,
        image_size: int = 224,
        max_samples: int = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_samples = max_samples
        
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self._hf_stream = None

    def _get_stream(self):
        if self._hf_stream is None:
            self._hf_stream = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=True,
            )
        return self._hf_stream

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch_example(self, iterator):
        """Fetch a single example with retry logic for transient network failures."""
        return next(iterator)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        stream = self._get_stream()
        iterator = iter(stream)
        count = 0
        
        while True:
            if self.max_samples and count >= self.max_samples:
                return
                
            try:
                example = self._fetch_example(iterator)
            except StopIteration:
                return
            except Exception as e:
                print(f"Error fetching example, retrying... {e}")
                continue

            try:
                img = example["image"]
                if not isinstance(img, Image.Image):
                    from PIL import Image as PILImage
                    img = PILImage.fromarray(img) if isinstance(img, np.ndarray) else PILImage.open(img)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_tensor = self.preprocess(img)
                
                labels = example.get("labels", example.get("label", 0))
                if isinstance(labels, list):
                    label = float(labels[0]) if labels else 0.0
                else:
                    label = float(labels)
                
                yield img_tensor, torch.tensor(label, dtype=torch.float32)
                count += 1
                
            except Exception as e:
                print(f"Error processing example: {e}")
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
    dataset_name: str = "alkzar90/NIH-Chest-X-ray-dataset",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_samples_per_epoch: int = 10000,
    val_samples: int = 2000,
    device: str = None,
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
    
    train_dataset = StreamedChestXrayDataset(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        max_samples=max_samples_per_epoch,
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    
    val_dataset = StreamedChestXrayDataset(
        dataset_name=dataset_name,
        split="validation",
        batch_size=batch_size,
        max_samples=val_samples,
    )
    val_loader = create_dataloader(val_dataset, batch_size=batch_size)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Max samples per epoch: {max_samples_per_epoch}")
    print(f"Validation samples: {val_samples}")
    
    best_val_loss = float('inf')
    
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
        "dataset": dataset_name,
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