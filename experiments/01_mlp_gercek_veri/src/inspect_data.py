from pathlib import Path

import pandas as pd


# inspect_data.py dosyasının bulunduğu klasör
SRC_DIR = Path(__file__).resolve().parent

# 01_mlp_gercek_veri klasörü
EXPERIMENT_DIR = SRC_DIR.parent

# Ham veri klasörü
RAW_DATA_DIR = EXPERIMENT_DIR / "data" / "raw"

# Veri dosyasının yolu
DATA_PATH = RAW_DATA_DIR /"breast+cancer+wisconsin+diagnostic" / "wdbc.data"


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


def main():
    print("Data path:", DATA_PATH)
    print("File exists:", DATA_PATH.exists())

    data = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
    )

    print("\nData shape:")
    print(data.shape)

    print("\nFirst five rows:")
    print(data.head())

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isna().sum())

    print("\nDiagnosis counts:")
    print(data["diagnosis"].value_counts())

    print("\nDiagnosis proportions:")
    print(data["diagnosis"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
