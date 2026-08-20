import os
import torch
import torch.onnx
from pathlib import Path

from models.imaging_inference import DenseNet121LSTM, ImagingInferenceEngine


def export_imaging_model_to_onnx(
    model_path: str = "models/imaging_densenet121_lstm.pth",
    onnx_path: str = "models/imaging_densenet121_lstm.onnx",
    quantized_onnx_path: str = "models/imaging_densenet121_lstm_int8.onnx",
    device: str = "cpu",
    input_shape: tuple = (1, 1, 3, 224, 224),
    opset_version: int = 17,
):
    """
    Export DenseNet121-LSTM imaging model to ONNX format with optional INT8 quantization.
    """
    print(f"Loading model from {model_path}...")
    engine = ImagingInferenceEngine(model_path=model_path, device=device)
    engine.load()
    model = engine.model
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    print(f"Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {onnx_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")

    # Try to apply INT8 quantization
    try:
        print("Applying INT8 quantization...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_onnx_path,
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized ONNX model saved to {quantized_onnx_path}")
    except ImportError:
        print("onnxruntime not available for quantization. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"Quantization failed: {e}")

    return onnx_path


def export_tabular_model_to_onnx(
    model_path: str = "models/tabular_xgb.pkl",
    onnx_path: str = "models/tabular_xgb.onnx",
    input_shape: tuple = (1, 13),
    opset_version: int = 15,
):
    """
    Export XGBoost tabular model to ONNX format using xgboost's native converter.
    """
    import joblib
    import xgboost as xgb
    from xgboost import XGBClassifier

    print(f"Loading XGBoost model from {model_path}...")
    model = joblib.load(model_path)

    # Get the base estimator if calibrated
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, 'estimator'):
        base_model = model.estimator
    else:
        base_model = model

    # Use XGBoost's native ONNX export
    print(f"Converting to ONNX: {onnx_path}...")
    try:
        # Try using xgboost's built-in ONNX export
        import onnx
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType(input_shape))]
        
        onnx_model = convert_xgboost(base_model, initial_types=initial_type, target_opset=opset_version)
        
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Tabular ONNX model saved to {onnx_path}")
    except ImportError:
        print("onnxmltools not available. Install with: pip install onnxmltools")
        # Fallback: try xgboost's experimental ONNX support
        try:
            from xgboost import to_onnx
            import numpy as np
            
            X_sample = np.random.randn(*input_shape).astype(np.float32)
            onnx_model = to_onnx(base_model, X_sample, target_opset=opset_version)
            
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"Tabular ONNX model saved to {onnx_path} (via xgboost.to_onnx)")
        except Exception as e:
            print(f"XGBoost ONNX export failed: {e}")
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["imaging", "tabular", "both"], default="both")
    parser.add_argument("--imaging-model", default="models/imaging_densenet121_lstm.pth")
    parser.add_argument("--tabular-model", default="models/tabular_xgb.pkl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.model_type in ["imaging", "both"]:
        if Path(args.imaging_model).exists():
            export_imaging_model_to_onnx(
                model_path=args.imaging_model,
                onnx_path="models/imaging_densenet121_lstm.onnx",
                device=args.device,
            )
        else:
            print(f"Imaging model not found at {args.imaging_model}, skipping...")

    if args.model_type in ["tabular", "both"]:
        if Path(args.tabular_model).exists():
            export_tabular_model_to_onnx(
                model_path=args.tabular_model,
                onnx_path="models/tabular_xgb.onnx",
            )
        else:
            print(f"Tabular model not found at {args.tabular_model}, skipping...")