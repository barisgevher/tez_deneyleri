from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------
# 1. Yollar ve sabitler
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SRC_DIR.parent

RAW_DATA_DIR = EXPERIMENT_DIR / "data" / "raw" / "breast+cancer+wisconsin+diagnostic"
DATA_PATH = RAW_DATA_DIR / "wdbc.data"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3



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


# --------------------------------------------------
# 2. Dataset sınıfı
# --------------------------------------------------

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
        return self.features[index], self.labels[index]


# --------------------------------------------------
# 3. Veriyi hazırla
# --------------------------------------------------

def create_dataloaders():
    data = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
    )

    # B = 0, M = 1
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = BreastCancerDataset(
        X_train_scaled,
        y_train,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


# --------------------------------------------------
# 4. MLP modeli
# --------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)

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

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy



# --------------------------------------------------
# 5. Modeli bir batch üzerinde çalıştır
# --------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, test_loader = create_dataloaders()

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    model = MLPClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Model parameters:")
    for name, parameter in model.named_parameters():
        print(
            name,
            tuple(parameter.shape),
            parameter.device,
        )

    print("\nTraining started...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        epoch_gradient_norms = []

        for batch_index, (inputs, labels) in enumerate(
            train_loader
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Önceki batch'in gradientlerini temizle
            optimizer.zero_grad()

            # 1. Forward pass
            logits = model(inputs)

            # 2. Loss hesapla
            loss = criterion(logits, labels)

            # 3. Gradient hesapla
            loss.backward()

            # Yalnızca ilk batch'te gradientleri incele
            if epoch == 1 and batch_index == 0:
                print("\nFirst training batch:")
                print("Inputs shape:", inputs.shape)
                print("Labels shape:", labels.shape)
                print("Logits shape:", logits.shape)
                print("Loss:", loss.item())

                print("\nGradients after backward:")

                for name, parameter in model.named_parameters():
                    print(
                        name,
                        "parameter shape:",
                        tuple(parameter.shape),
                        "gradient shape:",
                        tuple(parameter.grad.shape),
                        "gradient norm:",
                        parameter.grad.norm().item(),
                    )

            # 4. Gradient büyüklüğünü ölç ve sınırla
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            epoch_gradient_norms.append(
                float(gradient_norm)
            )

            # 5. Parametreleri GÜNCELLE
            # Bu satır batch başına yalnızca bir kez çalışır
            optimizer.step()

            # 6. Eğitim istatistiklerini topla
            batch_size = labels.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            predictions = logits.argmax(dim=1)

            total_train_correct += (
                predictions == labels
            ).sum().item()

            total_train_examples += batch_size

        # Epoch eğitim sonuçları
        train_loss = (
            total_train_loss / total_train_examples
        )

        train_accuracy = (
            total_train_correct / total_train_examples
        )

        average_gradient_norm = (
            sum(epoch_gradient_norms)
            / len(epoch_gradient_norms)
        )

        # Test değerlendirmesi
        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_accuracy:.4f} | "
            f"grad_norm={average_gradient_norm:.4f}"
        )


if __name__ == "__main__":
    main()
