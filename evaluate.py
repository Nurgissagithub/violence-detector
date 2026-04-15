import argparse
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset import ViolenceDataset
from model import ViolenceClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--weights", default="models/violence_classifier.safetensors")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = ViolenceDataset(args.data_root, split="val", num_frames=args.num_frames)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ViolenceClassifier(num_frames=args.num_frames).to(device)
    state_dict = load_file(args.weights)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device)
            logits = model(clips)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)    
    print(f"\nValidation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["NonViolence", "Violence"]))

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["NonViolence", "Violence"],
                yticklabels=["NonViolence", "Violence"])
    plt.title("Confusion Matrix")
    plt.savefig("CM.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()