import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from train_real_mlp import (
    BEST_CHECKPOINT_PATH,
    create_dataloaders,
    MLPClassifier,
    set_seed,
)


# --------------------------------------------------
# Deney yolları
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SRC_DIR.parent

RESULTS_DIR = EXPERIMENT_DIR / "results"
EVALUATION_JSON_PATH = RESULTS_DIR / "evaluation_results.json"
REPORT_PATH = RESULTS_DIR / "experiment_report.md"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# train_real_mlp.py içindeki deney ayarları
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# --------------------------------------------------
# Tahminleri toplama
# --------------------------------------------------

def collect_predictions(model, data_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    return all_labels, all_predictions, all_probabilities


# --------------------------------------------------
# Specificity hesaplama
# --------------------------------------------------

def calculate_specificity(confusion):
    # confusion = [[TN, FP], [FN, TP]]
    true_negative = confusion[0][0]
    false_positive = confusion[0][1]

    denominator = true_negative + false_positive

    if denominator == 0:
        return 0.0

    return true_negative / denominator


# --------------------------------------------------
# Markdown raporu
# --------------------------------------------------

def write_markdown_report(
    report_path,
    checkpoint,
    split_info,
    confusion,
    report_dict,
    accuracy,
    specificity,
):
    true_negative = confusion[0][0]
    false_positive = confusion[0][1]
    false_negative = confusion[1][0]
    true_positive = confusion[1][1]

    benign = report_dict["benign"]
    malignant = report_dict["malignant"]

    macro_avg = report_dict["macro avg"]
    weighted_avg = report_dict["weighted avg"]

    report = f"""# Breast Cancer Wisconsin Diagnostic — MLP Deney Raporu

## 1. Deney özeti

Bu deneyde UCI Breast Cancer Wisconsin Diagnostic veri seti üzerinde çok katmanlı algılayıcı (MLP) tabanlı ikili sınıflandırma gerçekleştirilmiştir.

Sınıf kodlaması:

- `0`: benign
- `1`: malignant

## 2. Veri seti

- Kaynak: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Toplam örnek: {split_info['total_samples']}
- Kullanılan özellik sayısı: 30
- Hedef değişken: `diagnosis`
- `B` → `0` → benign
- `M` → `1` → malignant
- Eksik değer: Veri hazırlama aşamasında eksik değer bulunmamıştır.

## 3. Veri bölme

| Bölüm | Örnek sayısı |
|---|---:|
| Eğitim | {split_info['train_samples']} |
| Validation | {split_info['validation_samples']} |
| Test | {split_info['test_samples']} |

Scaler yalnızca eğitim verisi üzerinde fit edilmiş; validation ve test verileri eğitimden öğrenilen scaler ile dönüştürülmüştür.

## 4. Model ve eğitim

- Model: MLP
- Mimari: `30 → 64 → 32 → 2`
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Batch size: {BATCH_SIZE}
- Learning rate: {LEARNING_RATE}
- Seed: {checkpoint['seed']}
- En iyi epoch: {checkpoint['epoch']}
- En iyi validation loss: {checkpoint['best_val_loss']:.6f}

Model seçimi validation loss kullanılarak yapılmış, test seti yalnızca nihai değerlendirme için kullanılmıştır.

## 5. Confusion matrix

Satırlar gerçek sınıfı, sütunlar model tahminini göstermektedir.

| Gerçek / Tahmin | Benign (0) | Malignant (1) |
|---|---:|---:|
| Benign (0) | {true_negative} | {false_positive} |
| Malignant (1) | {false_negative} | {true_positive} |

- TN — benign doğru sınıflandırılan: {true_negative}
- FP — benign olup malignant tahmin edilen: {false_positive}
- FN — malignant olup benign tahmin edilen: {false_negative}
- TP — malignant doğru sınıflandırılan: {true_positive}

## 6. Sınıf bazlı metrikler

| Sınıf | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Benign (0) | {benign['precision']:.4f} | {benign['recall']:.4f} | {benign['f1-score']:.4f} | {int(benign['support'])} |
| Malignant (1) | {malignant['precision']:.4f} | {malignant['recall']:.4f} | {malignant['f1-score']:.4f} | {int(malignant['support'])} |
| Macro average | {macro_avg['precision']:.4f} | {macro_avg['recall']:.4f} | {macro_avg['f1-score']:.4f} | {int(macro_avg['support'])} |
| Weighted average | {weighted_avg['precision']:.4f} | {weighted_avg['recall']:.4f} | {weighted_avg['f1-score']:.4f} | {int(weighted_avg['support'])} |

## 7. Genel sonuçlar

- Accuracy: **{accuracy:.4f}** ({accuracy * 100:.2f}%)
- Benign specificity: **{specificity:.4f}**
- Malignant precision: **{malignant['precision']:.4f}**
- Malignant recall: **{malignant['recall']:.4f}**
- Malignant F1-score: **{malignant['f1-score']:.4f}**

Malignant recall, gerçek malignant örneklerin ne kadarının doğru biçimde malignant olarak tespit edildiğini gösterir. False negative değeri, malignant olduğu halde benign tahmin edilen örnek sayısını gösterir.

## 8. Sınırlılıklar

- Sonuçlar tek bir random seed (`{checkpoint['seed']}`) ile elde edilmiştir.
- Veri seti küçüktür ve test seti {split_info['test_samples']} örnekten oluşmaktadır.
- Tek bir train/validation/test bölmesi kullanılmıştır.
- Çoklu seed ve k-fold cross-validation henüz uygulanmamıştır.
- Model karşılaştırması ve hiperparametre taraması henüz yapılmamıştır.
- Bu çalışma klinik karar veya tıbbi kullanım amacı taşımaz.

## 9. Sonraki adımlar

1. Farklı random seed değerleriyle tekrarlı deney.
2. Stratified k-fold cross-validation.
3. Logistic Regression ve Random Forest baseline modelleri.
4. MLP mimari ve hiperparametre karşılaştırması.
5. ROC-AUC ve PR-AUC analizi.
6. Confusion matrix görselleştirmesi.
"""

    report_path.write_text(report, encoding="utf-8")


# --------------------------------------------------
# Ana değerlendirme
# --------------------------------------------------

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", BEST_CHECKPOINT_PATH)
    print("Checkpoint exists:", BEST_CHECKPOINT_PATH.exists())

    if not BEST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"En iyi checkpoint bulunamadı: {BEST_CHECKPOINT_PATH}"
        )

    _, _, test_loader, split_info = create_dataloaders()

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = MLPClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels, predictions, probabilities = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    confusion = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).tolist()

    report_dict = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["benign", "malignant"],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["benign", "malignant"],
        zero_division=0,
    )

    accuracy = float(accuracy_score(labels, predictions))
    specificity = float(calculate_specificity(confusion))

    evaluation_results = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["best_val_loss"]),
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "specificity_benign": specificity,
        "classification_report": report_dict,
        "true_labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "split_info": split_info,
        "device": str(device),
    }

    EVALUATION_JSON_PATH.write_text(
        json.dumps(evaluation_results, indent=2),
        encoding="utf-8",
    )

    report_checkpoint = {
        "epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "seed": SEED,
    }

    write_markdown_report(
        report_path=REPORT_PATH,
        checkpoint=report_checkpoint,
        split_info=split_info,
        confusion=confusion,
        report_dict=report_dict,
        accuracy=accuracy,
        specificity=specificity,
    )

    print("\nConfusion matrix:")
    print(confusion)

    print("\nClassification report:")
    print(report_text)

    print("Accuracy:", f"{accuracy:.4f}")
    print("Benign specificity:", f"{specificity:.4f}")
    print("\nEvaluation JSON:")
    print(EVALUATION_JSON_PATH)
    print("\nExperiment report:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
