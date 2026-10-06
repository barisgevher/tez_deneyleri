import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ==================================================
# 1. Deney yolları ve ayarlar
# ==================================================

SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SRC_DIR.parent

RAW_DATA_DIR = EXPERIMENT_DIR / "data" / "raw" / "breast+cancer+wisconsin+diagnostic"
LOG_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENT_DIR / "results"

DATA_PATH = RAW_DATA_DIR / "wdbc.data"
HISTORY_PATH = LOG_DIR / "mlp_history.json"
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "mlp_best.pt"
)
FINAL_RESULTS_PATH = (
    RESULTS_DIR / "final_results.json"
)

for directory in [
    LOG_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

PATIENCE = 8
MIN_DELTA = 1e-4

INPUT_DIM = 30
NUM_CLASSES = 2


# ==================================================
# 2. Veri seti sütunları
# ==================================================

COLUMNS = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


# ==================================================
# 3. Tekrarlanabilirlik
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================
# 4. Dataset
# ==================================================

class BreastCancerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(
            features,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels.to_numpy(),
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.labels[index],
        )


# ==================================================
# 5. Veri okuma, üçe ayırma ve standardizasyon
# ==================================================

def create_dataloaders():
    data = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
    )

    data["target"] = data["diagnosis"].map(
        {
            "B": 0,
            "M": 1,
        }
    )

    feature_columns = [
        column
        for column in COLUMNS
        if column not in ["id", "diagnosis"]
    ]

    X = data[feature_columns]
    y = data["target"]

    # Önce test setini ayırıyoruz.
    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=SEED,
            stratify=y,
        )
    )

    # Train/validation ayrımı.
    # train_val'ın %20'si validation olur.
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.20,
            random_state=SEED,
            stratify=y_train_val,
        )
    )

    # Kritik nokta:
    # Scaler yalnızca train verisinde fit edilir.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = BreastCancerDataset(
        X_train_scaled,
        y_train,
    )

    val_dataset = BreastCancerDataset(
        X_val_scaled,
        y_val,
    )

    test_dataset = BreastCancerDataset(
        X_test_scaled,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    split_info = {
        "total_samples": len(data),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_class_counts": {
            str(key): int(value)
            for key, value in y_train.value_counts().items()
        },
        "validation_class_counts": {
            str(key): int(value)
            for key, value in y_val.value_counts().items()
        },
        "test_class_counts": {
            str(key): int(value)
            for key, value in y_test.value_counts().items()
        },
    }

    return (
        train_loader,
        val_loader,
        test_loader,
        split_info,
    )


# ==================================================
# 6. MLP modeli
# ==================================================

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


# ==================================================
# 7. Değerlendirme
# ==================================================

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


# ==================================================
# 8. Ana eğitim akışı
# ==================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Data path:", DATA_PATH)
    print("Data exists:", DATA_PATH.exists())

    (
        train_loader,
        val_loader,
        test_loader,
        split_info,
    ) = create_dataloaders()

    print("\nDataset split:")
    print(json.dumps(split_info, indent=2))

    model = MLPClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nTraining started...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0
        gradient_norms = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            logits = model(inputs)

            # Loss
            loss = criterion(logits, labels)

            # Backward
            loss.backward()

            # Gradient clipping
            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
            )

            gradient_norms.append(
                float(gradient_norm)
            )

            # Parametre güncellemesi
            optimizer.step()

            batch_size = labels.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            predictions = logits.argmax(dim=1)

            total_train_correct += (
                predictions == labels
            ).sum().item()

            total_train_examples += batch_size

        train_loss = (
            total_train_loss
            / total_train_examples
        )

        train_accuracy = (
            total_train_correct
            / total_train_examples
        )

        average_gradient_norm = (
            sum(gradient_norms)
            / len(gradient_norms)
        )

        # Validation: ağırlık güncellemesi yapılmaz
        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "gradient_norm": average_gradient_norm,
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"grad_norm={average_gradient_norm:.4f}"
        )

        # En iyi validation loss kontrolü
        improved = (
            best_val_loss - val_loss
            > MIN_DELTA
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "seed": SEED,
                    "input_dim": INPUT_DIM,
                    "num_classes": NUM_CLASSES,
                    "history": history,
                    "split_info": split_info,
                },
                BEST_CHECKPOINT_PATH,
            )

            print(
                f"  Best checkpoint saved "
                f"(epoch={epoch}, "
                f"val_loss={val_loss:.6f})"
            )

        else:
            epochs_without_improvement += 1

            print(
                "  No validation improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                "\nEarly stopping triggered."
            )
            print(
                f"Best epoch: {best_epoch}"
            )
            print(
                f"Best validation loss: "
                f"{best_val_loss:.6f}"
            )
            break

    # History dosyasını kaydet
    HISTORY_PATH.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("\nHistory saved to:")
    print(HISTORY_PATH)

    # En iyi checkpoint'i yükle
    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Test yalnızca burada kullanılıyor
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    final_results = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint[
            "best_val_loss"
        ],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "device": str(device),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "split_info": split_info,
        "history_path": str(HISTORY_PATH),
        "best_checkpoint_path": str(
            BEST_CHECKPOINT_PATH
        ),
    }

    FINAL_RESULTS_PATH.write_text(
        json.dumps(final_results, indent=2),
        encoding="utf-8",
    )

    print("\nFinal evaluation using best checkpoint:")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nResults saved to:")
    print(FINAL_RESULTS_PATH)


if __name__ == "__main__":
    main()
