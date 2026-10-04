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


# --------------------------------------------------
# 5. Modeli bir batch üzerinde çalıştır
# --------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, test_loader = create_dataloaders()

    model = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer eklendi artık Model parametreleri optimizer a veriyor
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    # Train loader'dan tek bir batch al
    inputs, labels = next(iter(train_loader))

    print("\nBefore device transfer:")
    print("Inputs shape:", inputs.shape)
    print("Labels shape:", labels.shape)
    print("Inputs dtype:", inputs.dtype)
    print("Labels dtype:", labels.dtype)

    # Veriyi GPU'ya veya CPU'ya taşı
    inputs = inputs.to(device)
    labels = labels.to(device)

    print("\nAfter device transfer:")
    print("Inputs device:", inputs.device)
    print("Labels device:", labels.device)

    # Forward pass
    logits = model(inputs)

    print("\nAfter forward pass:")
    print("Logits shape:", logits.shape)
    print("Logits device:", logits.device)

    # İlk örneğin logits değeri
    print("First logits:", logits[0].detach().cpu())

    # Olasılıklar yalnızca inceleme amacıyla
    probabilities = torch.softmax(logits, dim=1)

    print(
        "First probabilities:",
        probabilities[0].detach().cpu(),
    )

    # Tahmin
    predictions = logits.argmax(dim=1)

    print("First prediction:", predictions[0].item())
    print("First true label:", labels[0].item())

    # Loss
    loss = criterion(logits, labels)

    print("\nLoss:")
    print(loss.item())

    # Gradient hesaplama
    # loss.backward()

    # print("\nGradients:")
    # for name, parameter in model.named_parameters():
    #     print(
    #         name,
    #         "parameter shape:",
    #         tuple(parameter.shape),
    #         "gradient shape:",
    #         tuple(parameter.grad.shape),
    #         "gradient norm:",
    #         parameter.grad.norm().item(),
    #     )
    loss.backward()

    print("\nGradients:")
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


        # optimizer.step() öncesi ağırlığın bir kopyasını al
    weight_before_step = (
            model.network[0].weight.detach().clone()
        )

        # Optimizer gradientleri kullanarak ağırlıkları günceller
    optimizer.step()
    with torch.no_grad():
            updated_logits = model(inputs)
            updated_loss = criterion(updated_logits, labels)

    print("Loss before optimizer.step():", loss.item())
    print("Loss after optimizer.step():", updated_loss.item())


        # optimizer.step() sonrası ağırlığı al
    weight_after_step = (
            model.network[0].weight.detach().clone()
        )

        # İki ağırlık tensorü arasındaki farkı hesapla
    weight_difference = (
            weight_after_step - weight_before_step
        ).abs()

    print("\nAfter optimizer.step():")
    print(
        "Maximum weight change:",
        weight_difference.max().item(),
    )
    print(
        "Mean weight change:",
        weight_difference.mean().item(),
    )
    print(
        "First five weights before:",
        weight_before_step[0, :5].cpu(),
    )
    print(
        "First five weights after:",
        weight_after_step[0, :5].cpu(),
    )



if __name__ == "__main__":
    main()
