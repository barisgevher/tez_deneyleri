from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SRC_DIR.parent

RAW_DATA_DIR = EXPERIMENT_DIR / "data" / "raw" / "breast+cancer+wisconsin+diagnostic"
DATA_PATH = RAW_DATA_DIR / "wdbc.data"


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


def load_data():
    return pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
    )


def main():
    data = load_data()

    # 1. Hedef etiketi sayısala dönüştür
    data["target"] = data["diagnosis"].map(
        {
            "B": 0,
            "M": 1,
        }
    )

    # 2. ID ve diagnosis dışındaki 30 sütunu seç
    feature_columns = [
        column
        for column in COLUMNS
        if column not in ["id", "diagnosis"]
    ]

    X = data[feature_columns]
    y = data["target"]

    print("Before split")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:")
    print(y.value_counts().sort_index())

    # 3. Train/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nAfter split")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTrain class counts:")
    print(y_train.value_counts().sort_index())

    print("\nTest class counts:")
    print(y_test.value_counts().sort_index())

    # 4. Scaler oluştur
    scaler = StandardScaler()

    # 5. Scaler SADECE train verisinde öğrenir
    scaler.fit(X_train)

    # 6. Train ve test verisini dönüştür
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled shapes")
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)

    print("\nTrain scaled statistics")
    print(
        "Train means, first 5 features:",
        X_train_scaled.mean(axis=0)[:5],
    )
    print(
        "Train stds, first 5 features:",
        X_train_scaled.std(axis=0)[:5],
    )

    print("\nTest scaled statistics")
    print(
        "Test means, first 5 features:",
        X_test_scaled.mean(axis=0)[:5],
    )
    print(
        "Test stds, first 5 features:",
        X_test_scaled.std(axis=0)[:5],
    )


if __name__ == "__main__":
    main()
