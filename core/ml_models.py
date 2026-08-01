"""
Optional real pretrained-model inference (ResNet101, Faster R-CNN).

If torch/torchvision aren't installed, or weights can't be downloaded (no
internet), the relevant panel says so plainly instead of fabricating
results. ImageNet/COCO classes are generic object categories, not crop
disease labels -- these panels are genuine, per-image plausibility/feature
checks, not validated agronomic pathology classifiers.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import torch
    from torchvision.models import resnet101, ResNet101_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_resnet_model, _resnet_weights, _resnet_error = None, None, None
_frcnn_model, _frcnn_weights, _frcnn_error = None, None, None


def _load_resnet() -> None:
    global _resnet_model, _resnet_weights, _resnet_error
    if _resnet_model is not None or _resnet_error is not None:
        return
    try:
        _resnet_weights = ResNet101_Weights.DEFAULT
        _resnet_model = resnet101(weights=_resnet_weights)
        _resnet_model.eval()
    except Exception as e:
        _resnet_error = str(e)


def _load_frcnn() -> None:
    global _frcnn_model, _frcnn_weights, _frcnn_error
    if _frcnn_model is not None or _frcnn_error is not None:
        return
    try:
        _frcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _frcnn_model = fasterrcnn_resnet50_fpn(weights=_frcnn_weights)
        _frcnn_model.eval()
    except Exception as e:
        _frcnn_error = str(e)


def _unavailable_panel(ax, title: str, detail: str) -> None:
    ax.axis("off")
    ax.text(0.02, 0.5, f"{title} unavailable in this environment.\n{detail}\n\nNo results were fabricated.",
            fontsize=8, va="center", family="monospace", wrap=True)


def resnet101_classification(image_array: np.ndarray, ax) -> None:
    """Real inference with the ImageNet-pretrained ResNet101 (torchvision)."""
    if not TORCH_AVAILABLE:
        _unavailable_panel(ax, "ResNet101", "Install with: pip install torch torchvision")
        return
    _load_resnet()
    if _resnet_model is None:
        _unavailable_panel(ax, "ResNet101",
                            f"Could not load pretrained weights ({_resnet_error}).\nUsually means no internet access to fetch them.")
        return
    img = Image.fromarray(image_array)
    tensor = _resnet_weights.transforms()(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(_resnet_model(tensor)[0], dim=0)
    top5 = torch.topk(probs, 5)
    labels = [_resnet_weights.meta["categories"][i] for i in top5.indices]
    scores = top5.values.numpy()
    ax.barh(labels[::-1], scores[::-1], color="teal")
    ax.set_xlabel("Confidence")
    ax.set_title("ResNet101 (ImageNet-pretrained, generic) — top 5")


def faster_rcnn_detection(image_array: np.ndarray, ax) -> None:
    """Real inference with the COCO-pretrained Faster R-CNN (torchvision)."""
    import matplotlib.pyplot as plt
    if not TORCH_AVAILABLE:
        _unavailable_panel(ax, "Faster R-CNN", "Install with: pip install torch torchvision")
        return
    _load_frcnn()
    if _frcnn_model is None:
        _unavailable_panel(ax, "Faster R-CNN",
                            f"Could not load pretrained weights ({_frcnn_error}).\nUsually means no internet access to fetch them.")
        return
    img = Image.fromarray(image_array)
    tensor = _frcnn_weights.transforms()(img)
    with torch.no_grad():
        pred = _frcnn_model([tensor])[0]
    ax.imshow(image_array)
    categories = _frcnn_weights.meta["categories"]
    count = 0
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < 0.5:
            continue
        count += 1
        x1, y1, x2, y2 = box.numpy()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="orange", linewidth=1.5))
        ax.text(x1, max(y1 - 3, 0), f"{categories[label]}: {score:.2f}", fontsize=7, color="orange")
    ax.set_title(f"Faster R-CNN (COCO-pretrained, generic): {count} detection(s) > 0.5")
    ax.axis("off")
