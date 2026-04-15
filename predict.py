import argparse
import torch
from safetensors.torch import load_file

from dataset import build_transforms, sample_frames
from model import ViolenceClassifier

LABELS = {0: "NonViolence", 1: "Violence"}


def predict_video(video_path: str, weights: str, num_frames: int = 16, threshold: float = 0.65) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViolenceClassifier(num_frames=num_frames).to(device)
    state_dict = load_file(weights)
    model.load_state_dict(state_dict)
    model.eval()

    transform = build_transforms(train=False)
    frames = sample_frames(video_path, num_frames)
    if not frames:
        return {"error": "Could not read video"}

    clip = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip)
        probs = torch.softmax(logits, dim=1)[0]

    violence_prob = probs[1].item()
    pred = 1 if violence_prob >= threshold else 0

    return {
        "prediction": LABELS[pred],
        "threshold_used": threshold,
        "confidence": f"{probs[pred].item():.4f}",
        "violence_prob": f"{violence_prob:.4f}",
        "non_violence_prob": f"{probs[0].item():.4f}",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="models/violence_classifier.safetensors")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.65)
    args = p.parse_args()

    result = predict_video(args.video, args.weights, args.num_frames, args.threshold)
    print(result)
