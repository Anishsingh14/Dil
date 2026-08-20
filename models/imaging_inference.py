import asyncio
import io
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageStat
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from scipy import ndimage, signal


class DenseNet121LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        
        self.features = backbone.features
        self.feature_dim = backbone.classifier.in_features
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        
        features = self.features(x)
        features = self.avgpool(features)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]
        
        logits = self.classifier(last_hidden)
        return torch.sigmoid(logits).squeeze(-1)


class ImagingInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/imaging_densenet121_lstm.pth",
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DenseNet121LSTM] = None
        self.model_version = "imaging-densenet121-lstm-v1.1.0"
        self._loaded = False
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        if self._loaded:
            return

        self.model = DenseNet121LSTM(
            num_classes=1,
            lstm_hidden=256,
            lstm_layers=1,
            dropout=0.3,
            pretrained=False,
        ).to(self.device)
        
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_version = checkpoint.get("model_version", self.model_version)
            print(f"Loaded imaging model from {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}, using ImageNet-pretrained DenseNet121 + random LSTM")
            self.model.features.load_state_dict(
                torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True).features.state_dict()
            )
        
        self.model.eval()
        self._loaded = True
        print(f"Imaging model loaded on {self.device}: {self.model_version}")

    def _load_image(self, file_bytes: bytes, filename: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Load image with proper VOI LUT pipeline for DICOM and image quality checks.
        Returns (PIL Image, quality_metrics_dict).
        """
        filename_lower = filename.lower()
        quality_metrics = {}
        
        if filename_lower.endswith(('.dicom', '.dcm')):
            dicom = pydicom.dcmread(io.BytesIO(file_bytes))
            
            # Apply VOI LUT pipeline (per DICOM standard)
            # This handles Window Center/Width, VOI LUT, and Modality LUT
            try:
                pixel_array = apply_voi_lut(dicom.pixel_array, dicom)
                quality_metrics["voi_lut_applied"] = True
            except Exception as e:
                warnings.warn(f"VOI LUT application failed, falling back to Window Center/Width: {e}")
                quality_metrics["voi_lut_applied"] = False
                pixel_array = dicom.pixel_array.astype(np.float32)
                
                # Fallback to Window Center/Width
                if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                    wc = float(dicom.WindowCenter[0] if isinstance(dicom.WindowCenter, pydicom.multival.MultiValue) else dicom.WindowCenter)
                    ww = float(dicom.WindowWidth[0] if isinstance(dicom.WindowWidth, pydicom.multival.MultiValue) else dicom.WindowWidth)
                    img_min = wc - ww / 2
                    img_max = wc + ww / 2
                    pixel_array = np.clip(pixel_array, img_min, img_max)
                    quality_metrics["windowing_applied"] = True
                else:
                    quality_metrics["windowing_applied"] = False
            
            # Normalize to 0-255
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
            pixel_array = (pixel_array * 255).astype(np.uint8)
            img = Image.fromarray(pixel_array, mode='L')
            
            # Store DICOM metadata for quality metrics
            quality_metrics["modality"] = getattr(dicom, 'Modality', 'UNKNOWN')
            quality_metrics["rows"] = getattr(dicom, 'Rows', 0)
            quality_metrics["columns"] = getattr(dicom, 'Columns', 0)
            quality_metrics["bits_allocated"] = getattr(dicom, 'BitsAllocated', 0)
            quality_metrics["bits_stored"] = getattr(dicom, 'BitsStored', 0)
            quality_metrics["pixel_spacing"] = getattr(dicom, 'PixelSpacing', None)
            
        else:
            img = Image.open(io.BytesIO(file_bytes))
            quality_metrics["voi_lut_applied"] = False
            quality_metrics["windowing_applied"] = False
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Image quality checks
        quality_metrics.update(self._check_image_quality(img))
        
        return img, quality_metrics

    def _check_image_quality(self, img: Image.Image) -> Dict[str, Any]:
        """
        Perform image quality checks on the loaded image.
        Returns dict with quality metrics and flags.
        """
        metrics = {}
        
        try:
            # Convert to grayscale for analysis
            if img.mode != 'L':
                gray = img.convert('L')
            else:
                gray = img
            
            # Convert to numpy array
            arr = np.array(gray, dtype=np.float32)
            
            # 1. Basic dimensions
            metrics["width"] = int(img.width)
            metrics["height"] = int(img.height)
            metrics["aspect_ratio"] = round(img.width / img.height, 2)
            
            # 2. Exposure / Brightness (mean pixel value)
            mean_val = float(np.mean(arr))
            metrics["mean_intensity"] = round(mean_val, 2)
            metrics["exposure_ok"] = bool(30 < mean_val < 225)  # Not too dark/bright
            
            # 2. Contrast (standard deviation)
            std_val = float(np.std(arr))
            metrics["contrast"] = round(std_val, 2)
            metrics["contrast_ok"] = bool(std_val > 15)  # Minimum contrast
            
            # 3. Sharpness (Laplacian variance)
            laplacian = ndimage.laplace(arr)
            sharpness = float(np.var(laplacian))
            metrics["sharpness"] = round(sharpness, 2)
            metrics["sharpness_ok"] = bool(sharpness > 100)  # Threshold for blur detection
            
            # 4. Dynamic range
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            metrics["dynamic_range"] = round(max_val - min_val, 2)
            metrics["dynamic_range_ok"] = bool((max_val - min_val) > 50)
            
            # 5. Saturation check (percentage of saturated pixels)
            saturated_low = np.sum(arr <= 5) / arr.size
            saturated_high = np.sum(arr >= 250) / arr.size
            metrics["saturation_low_pct"] = round(saturated_low * 100, 2)
            metrics["saturation_high_pct"] = round(saturated_high * 100, 2)
            metrics["saturation_ok"] = bool(saturated_low < 0.05 and saturated_high < 0.05)
            
            # 6. Noise estimation (median absolute deviation of high-frequency components)
            # High-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_freq = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
            noise_est = float(np.median(np.abs(high_freq)))
            metrics["noise_estimate"] = round(noise_est, 2)
            metrics["noise_ok"] = bool(noise_est < 30)
            
            # Overall quality score
            checks = [
                bool(metrics.get("exposure_ok", False)),
                bool(metrics.get("contrast_ok", False)),
                bool(metrics.get("sharpness_ok", False)),
                bool(metrics.get("dynamic_range_ok", False)),
                bool(metrics.get("saturation_ok", False)),
                bool(metrics.get("noise_ok", False)),
            ]
            metrics["quality_checks_passed"] = int(sum(checks))
            metrics["quality_checks_total"] = int(len(checks))
            metrics["quality_score"] = round(sum(checks) / len(checks), 2)
            metrics["quality_ok"] = bool(all(checks))
            
        except Exception as e:
            metrics["quality_error"] = str(e)
            metrics["quality_ok"] = False
        
        return metrics

    def _predict_sync(self, file_bytes: bytes, filename: str) -> Tuple[float, str, dict]:
        if not self._loaded:
            self.load()

        img, quality_metrics = self._load_image(file_bytes, filename)
        img_tensor = self.preprocess(img).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 3, 224, 224]
        
        with torch.no_grad():
            risk_score = float(self.model(img_tensor).cpu().item())

        if risk_score < 0.33:
            risk_level = "Low Risk"
        elif risk_score < 0.66:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        findings = {
            "st_segment_abnormality_detected": risk_score > 0.5,
            "structural_risk_pattern_confidence": round(risk_score, 2),
            "image_quality": quality_metrics,
        }

        return risk_score, risk_level, findings

    async def predict(self, file_bytes: bytes, filename: str) -> Tuple[float, str, dict, int]:
        start_time = time.perf_counter()
        risk_score, risk_level, findings = await asyncio.to_thread(self._predict_sync, file_bytes, filename)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return risk_score, risk_level, findings, latency_ms

    async def predict_series(self, files_data: List[Tuple[bytes, str]]) -> Tuple[float, str, dict, int, List[float]]:
        """Process multiple images as a series for cardiac MRI or multi-view X-rays."""
        start_time = time.perf_counter()
        
        if not self._loaded:
            self.load()
        
        individual_scores = []
        all_quality_metrics = []
        
        # Process each image
        for file_bytes, filename in files_data:
            img, quality_metrics = self._load_image(file_bytes, filename)
            img_tensor = self.preprocess(img).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                score = float(self.model(img_tensor).cpu().item())
            
            individual_scores.append(score)
            all_quality_metrics.append(quality_metrics)
        
        # Aggregate scores (mean for series)
        risk_score = float(np.mean(individual_scores))
        
        if risk_score < 0.33:
            risk_level = "Low Risk"
        elif risk_score < 0.66:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
        
        # Aggregate quality metrics (worst case)
        all_quality_ok = all(m.get("quality_ok", False) for _, m in zip(files_data, all_quality_metrics))
        
        findings = {
            "st_segment_abnormality_detected": risk_score > 0.5,
            "structural_risk_pattern_confidence": round(risk_score, 2),
            "image_quality": {
                "individual": all_quality_metrics,
                "series_quality_ok": all_quality_ok,
            },
            "individual_scores": individual_scores,
        }
        
        if risk_score < 0.33:
            risk_level = "Low Risk"
        elif risk_score < 0.66:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return float(np.mean(individual_scores)), risk_level, findings, latency_ms, individual_scores

    def is_loaded(self) -> bool:
        return self._loaded

    def get_device(self) -> str:
        return str(self.device)


_imaging_engine: Optional[ImagingInferenceEngine] = None


def get_imaging_engine() -> ImagingInferenceEngine:
    global _imaging_engine
    if _imaging_engine is None:
        _imaging_engine = ImagingInferenceEngine()
    return _imaging_engine


def _check_image_quality(self, img: Image.Image) -> Dict[str, Any]:
        """
        Perform image quality checks on the loaded image.
        Returns dict with quality metrics and flags.
        """
        metrics = {}
        
        try:
            # Convert to grayscale for analysis
            if img.mode != 'L':
                gray = img.convert('L')
            else:
                gray = img
            
            # Convert to numpy array
            arr = np.array(gray, dtype=np.float32)
            
            # 1. Basic dimensions
            metrics["width"] = img.width
            metrics["height"] = img.height
            metrics["aspect_ratio"] = round(img.width / img.height, 2)
            
            # 2. Exposure / Brightness (mean pixel value)
            mean_val = float(np.mean(arr))
            metrics["mean_intensity"] = round(mean_val, 2)
            metrics["exposure_ok"] = 30 < mean_val < 225  # Not too dark/bright
            
            # 3. Contrast (standard deviation)
            std_val = float(np.std(arr))
            metrics["contrast"] = round(std_val, 2)
            metrics["contrast_ok"] = std_val > 15  # Minimum contrast
            
            # 3. Sharpness (Laplacian variance)
            laplacian = ndimage.laplace(arr)
            sharpness = float(np.var(laplacian))
            metrics["sharpness"] = round(sharpness, 2)
            metrics["sharpness_ok"] = sharpness > 100  # Threshold for blur detection
            
            # 4. Dynamic range
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            metrics["dynamic_range"] = round(max_val - min_val, 2)
            metrics["dynamic_range_ok"] = (max_val - min_val) > 50
            
            # 5. Saturation check (percentage of saturated pixels)
            saturated_low = np.sum(arr <= 5) / arr.size
            saturated_high = np.sum(arr >= 250) / arr.size
            metrics["saturation_low_pct"] = round(saturated_low * 100, 2)
            metrics["saturation_high_pct"] = round(saturated_high * 100, 2)
            metrics["saturation_ok"] = saturated_low < 0.05 and saturated_high < 0.05
            
            # 6. Noise estimation (median absolute deviation of high-frequency components)
            # High-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_freq = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
            noise_est = float(np.median(np.abs(high_freq)))
            metrics["noise_estimate"] = round(noise_est, 2)
            metrics["noise_ok"] = noise_est < 30
            
            # Overall quality score
            checks = [
                metrics.get("exposure_ok", False),
                metrics.get("contrast_ok", False),
                metrics.get("sharpness_ok", False),
                metrics.get("dynamic_range_ok", False),
                metrics.get("saturation_ok", False),
                metrics.get("noise_ok", False),
            ]
            metrics["quality_checks_passed"] = sum(checks)
            metrics["quality_checks_total"] = len(checks)
            metrics["quality_score"] = round(sum(checks) / len(checks), 2)
            metrics["quality_ok"] = all(checks)
            
        except Exception as e:
            metrics["quality_error"] = str(e)
            metrics["quality_ok"] = False
        
        return metrics


def initialize_imaging_model() -> None:
    engine = get_imaging_engine()
    engine.load()