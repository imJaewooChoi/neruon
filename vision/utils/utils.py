import yaml
import torch


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def export_onnx(model, dummy_tensor, onnx_path):
    torch.onnx.export(
        model,
        dummy_tensor,
        onnx_path,
    )

    print(f"Model exported to {onnx_path}")


def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
