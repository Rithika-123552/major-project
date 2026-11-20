"""RoadGuard AI Streamlit application."""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

from hybrid_model import CLASS_NAMES, build_hybrid_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "hybrid_road_damage.pt"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE = transforms.Normalize(
    models.ResNet50_Weights.DEFAULT.transforms().mean,
    models.ResNet50_Weights.DEFAULT.transforms().std,
)
PREPROCESS = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), NORMALIZE]
)
DESCRIPTIONS = {
    "Pothole": "A localized depression or loss of road surface that may require repair.",
    "Crack": "A visible linear fracture or network of fractures in the road surface.",
    "Asphalt": "A generally intact asphalt surface without a dominant visible defect.",
}


@st.cache_resource
def load_model(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hybrid checkpoint not found: {path}. "
            "Run train_hybrid.py before launching the application."
        )
    model = build_hybrid_model(
        checkpoint_path=path,
        pretrained=False,
        device=DEVICE,
    )
    model.eval()
    return model


def predict(model, image: Image.Image) -> tuple[str, np.ndarray, np.ndarray]:
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return CLASS_NAMES[int(probabilities.argmax())], probabilities, tensor


def grad_cam(model, tensor: torch.Tensor, class_index: int) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.cnn_backbone[-1]

    def save_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, class_index].backward()
    finally:
        handle.remove()
    if not activations or not gradients:
        raise RuntimeError("Grad-CAM could not capture CNN activations.")
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    heatmap = (weights * activations[0]).sum(dim=1).squeeze(0).relu()
    heatmap = heatmap / heatmap.max().clamp_min(1e-8)
    return heatmap.detach().cpu().numpy()


def render_grad_cam(image: Image.Image, heatmap: np.ndarray) -> bytes:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.imshow(image.convert("RGB"))
    axis.imshow(
        Image.fromarray(np.uint8(heatmap * 255)).resize(image.size),
        cmap="jet",
        alpha=0.42,
    )
    axis.axis("off")
    output = io.BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return output.getvalue()


def make_report(image: Image.Image, result: dict) -> bytes:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    payload = {
        "project": "RoadGuard AI",
        "analysis_time_utc": result["analysis_time_utc"],
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "explanation": result["explanation"],
        "image_base64_jpeg": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        "disclaimer": (
            "Research output only; not a certified road-safety assessment."
        ),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
    st.title("RoadGuard AI")
    st.caption("HYBRID CNN–VISION TRANSFORMER ROAD ANALYSIS")
    st.write("Upload a road image to identify its surface condition.")

    with st.sidebar:
        st.header("Model")
        checkpoint = st.text_input(
            "Hybrid checkpoint",
            str(DEFAULT_CHECKPOINT),
        )
        st.info(f"Running on {DEVICE}.")
        if st.button("Reset analysis"):
            st.session_state.pop("analysis", None)
            st.session_state.pop("history", None)
            st.rerun()

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="JPG, JPEG or PNG",
    )
    if uploaded is None:
        st.info("Upload a road image to begin analysis.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Selected road image", use_container_width=True)
    if not st.button("Analyse Road Image", type="primary"):
        return

    try:
        model = load_model(checkpoint)
        predicted, probabilities, tensor = predict(model, image)
        heatmap = grad_cam(model, tensor, CLASS_NAMES.index(predicted))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    result = {
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "predicted_class": predicted,
        "confidence": float(probabilities.max()),
        "class_probabilities": {
            name: float(probability)
            for name, probability in zip(CLASS_NAMES, probabilities)
        },
        "explanation": DESCRIPTIONS[predicted],
    }
    st.session_state["analysis"] = result
    st.session_state.setdefault("history", []).append(result)

    st.subheader("Analysis result")
    result_column, probability_column = st.columns(2)
    with result_column:
        st.metric("Predicted road class", predicted)
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.write(result["explanation"])
    with probability_column:
        st.write("Class probabilities")
        for name, probability in result["class_probabilities"].items():
            st.write(f"**{name}** — {probability * 100:.1f}%")
            st.progress(probability)

    st.subheader("Explainable analysis")
    st.image(
        render_grad_cam(image, heatmap),
        caption="Grad-CAM: regions influencing the CNN branch",
        use_container_width=True,
    )

    history = st.session_state.get("history", [])
    st.subheader("Analysis dashboard")
    dashboard_columns = st.columns(4)
    dashboard_columns[0].metric("Images analysed", len(history))
    for column, name in zip(dashboard_columns[1:], CLASS_NAMES):
        column.metric(
            f"{name} predictions",
            sum(item["predicted_class"] == name for item in history),
        )

    st.download_button(
        "Download analysis report",
        data=make_report(image, result),
        file_name="roadguard_analysis.json",
        mime="application/json",
    )
    st.caption(
        "Research output only; this system is not a certified road-safety assessment."
    )


if __name__ == "__main__":
    main()
