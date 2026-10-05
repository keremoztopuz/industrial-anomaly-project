"""Load the category models once at startup."""

from anomaly.model.patchcore import PatchCore


def load_models(directory):
    models, backbone = {}, None
    for path in sorted(directory.glob("*.pt")):
        model = PatchCore.load(path, device="cpu", backbone=backbone)
        backbone = model.backbone
        models[path.stem] = model
        print(f"Loaded model {path.stem} from {path}")
    if not models:
        raise RuntimeError(f"No category models (*.pt) found in MODEL_DIR={directory}")
    return models
