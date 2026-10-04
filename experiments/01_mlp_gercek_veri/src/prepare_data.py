from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SRC_DIR.parent
RAW_DATA_DIR = EXPERIMENT_DIR / "data" /  "raw" / "breast+cancer+wisconsin+diagnostic" 
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
    data = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
    )

    return data


def main():
    data = load_data()

    print("Original shape:", data.shape)

    # B = benign → 0
    # M = malignant → 1
    data["target"] = data["diagnosis"].map(
        {
            "B": 0,
            "M": 1,
        }
    )

    # ID ve metin biçimindeki diagnosis modele verilmeyecek
    feature_columns = [
        column
        for column in COLUMNS
        if column not in ["id", "diagnosis"]
    ]

    X = data[feature_columns]
    y = data["target"]

    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget values:")
    print(y.value_counts().sort_index())

    print("\nTarget value counts with labels:")
    print(data["diagnosis"].value_counts())

    print("\nFeature data types:")
    print(X.dtypes.value_counts())

    print("\nTarget data type:")
    print(y.dtype)

    print("\nMissing values in X:", X.isna().sum().sum())
    print("Missing values in y:", y.isna().sum())

    print("\nFirst feature row:")
    print(X.iloc[0])

    print("\nFirst target value:")
    print(y.iloc[0])

    print("\nFeature summary:")
    print(X.describe().T[["min", "max", "mean", "std"]])


if __name__ == "__main__":
    main()
