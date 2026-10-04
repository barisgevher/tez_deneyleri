from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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
        feature = self.features[index]
        label = self.labels[index]

        return feature, label


def load_and_prepare_data():
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

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )


def main():
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_and_prepare_data()

    train_dataset = BreastCancerDataset(
        X_train,
        y_train,
    )

    test_dataset = BreastCancerDataset(
        X_test,
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

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    print("Number of train batches:", len(train_loader))
    print("Number of test batches:", len(test_loader))

    first_feature, first_label = train_dataset[0]

    print("\nSingle sample:")
    print("Feature shape:", first_feature.shape)
    print("Label shape:", first_label.shape)
    print("Label value:", first_label.item())
    print("Feature dtype:", first_feature.dtype)
    print("Label dtype:", first_label.dtype)

    first_batch_features, first_batch_labels = next(
        iter(train_loader)
    )

    print("\nFirst train batch:")
    print("Features shape:", first_batch_features.shape)
    print("Labels shape:", first_batch_labels.shape)
    print("Features dtype:", first_batch_features.dtype)
    print("Labels dtype:", first_batch_labels.dtype)


if __name__ == "__main__":
    main()
