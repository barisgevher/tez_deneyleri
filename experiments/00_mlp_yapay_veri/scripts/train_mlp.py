import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_DIR = Path(__file__).resolve().parent

LOG_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENT_DIR / "results"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
INPUT_DIM = 20
HIDDEN_DIM = 64
NUM_CLASSES = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset():
    generator = torch.Generator().manual_seed(SEED)

    x = torch.randn(
        4000,
        INPUT_DIM,
        generator=generator,
    )

    signal = x[:, :5].sum(dim=1) - x[:, 5:10].sum(dim=1)
    y = (signal > 0).long()

    return TensorDataset(x, y)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def calculate_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = create_dataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MLPClassifier().to(device)


    # parametreleri yazdırma kodu 
    for name, parameter in model.named_parameters():
        print(
            name,
            parameter.shape,
            parameter.requires_grad,
            parameter.device,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    history = []

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        #for inputs, labels in loader:
        for batch_index, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            # logits inspection
            if epoch == 1 and batch_index == 0:
                print("Input shape:", inputs.shape)
                print("Labels shape:", labels.shape)
                print("Logits shape:", logits.shape)
                print("Input device:", inputs.device)
                print("Labels device:", labels.device)
                print("Logits device:", logits.device)
                print("First input:", inputs[0])
                print("First label:", labels[0])
                print("First logits:", logits[0])

                probabilities = torch.softmax(logits, dim=1)

                print(
                    "First probabilities:",
                    probabilities[0].detach().cpu()
                )

                print(
                    "First prediction:",
                    probabilities[0].argmax().item()
                )
            
            

            loss = criterion(logits, labels)
            loss.backward()
            global_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, labels)
            total_batches += 1

        epoch_loss = total_loss / total_batches
        epoch_accuracy = total_accuracy / total_batches

        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "gradient_norm": float(global_grad_norm),
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={epoch_loss:.4f} | "
            f"accuracy={epoch_accuracy:.4f} | "
            f"grad_norm={float(global_grad_norm):.4f}"
        )
        
    history_path = LOG_DIR / "mlp_history.json"
    checkpoint_path = CHECKPOINT_DIR / "mlp_final.pt"

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": SEED,
            "epoch": EPOCHS,
            "history": history,
        },
        checkpoint_path,
    )

    print("Training completed.")
    print(f"History: {history_path}")
    print(f"Checkpoint: {checkpoint_path}")







if __name__ == "__main__":
    main()