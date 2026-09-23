"""Treina uma primeira referência para os resultados H, D e A."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data/processed/matches_features_2020_2024.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "logistic_baseline_2020_2023.joblib"
METRICS_PATH = ARTIFACTS_DIR / "logistic_baseline_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/predictions_2024.csv"
VALIDATION_PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/predictions_2023_validation.csv"
VALIDATION_DIAGNOSTICS_PATH = ARTIFACTS_DIR / "validation_2023_diagnostics.json"

FEATURE_NAMES = (
    "pontos_por_jogo_ultimos_5",
    "media_gols_pro_ultimos_5",
    "media_gols_contra_ultimos_5",
    "pontos_por_jogo_mesmo_mando_ultimos_5",
    "media_gols_pro_mesmo_mando_ultimos_5",
    "media_gols_contra_mesmo_mando_ultimos_5",
)
FEATURE_COLUMNS = tuple(
    f"{side}_{feature}"
    for side in ("mandante", "visitante")
    for feature in FEATURE_NAMES
)
CLASSES = ("A", "D", "H")


def load_feature_table(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Carrega e verifica a base que saiu do notebook de features."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Base de features não encontrada: {path}. "
            "Execute todas as células de notebooks/02_feature_engineering.ipynb."
        )

    matches = pd.read_csv(path)
    required = {"id", "temporada", "data", "mandante", "visitante", "resultado"}
    required.update(FEATURE_COLUMNS)
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Colunas necessárias ausentes: {', '.join(sorted(missing))}")
    if not matches["id"].is_unique or matches["id"].isna().any():
        raise ValueError("Cada partida deve ter um ID único e preenchido.")
    if matches["resultado"].isna().any() or not matches["resultado"].isin(CLASSES).all():
        raise ValueError("Resultado deve ser H, D ou A em todas as partidas.")
    for column in FEATURE_COLUMNS:
        matches[column] = pd.to_numeric(matches[column], errors="raise")
    return matches


def split_by_season(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserva 2023 para validação e 2024 para a avaliação final."""
    expected_counts = {year: 380 for year in range(2020, 2025)}
    counts = matches["temporada"].value_counts().sort_index().to_dict()
    if counts != expected_counts:
        raise ValueError(f"Esperadas 380 partidas por temporada de 2020 a 2024; encontrado {counts}.")

    train = matches[matches["temporada"].isin((2020, 2021, 2022))].copy()
    validation = matches[matches["temporada"] == 2023].copy()
    test = matches[matches["temporada"] == 2024].copy()
    return train, validation, test


def make_model():
    """Aprende imputação, escala e regressão apenas no conjunto usado em fit."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )


def evaluate(model, matches: pd.DataFrame) -> dict[str, float]:
    """Mede acerto, equilíbrio entre classes e qualidade das probabilidades."""
    features = matches.loc[:, FEATURE_COLUMNS]
    actual = matches["resultado"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "f1_macro": float(f1_score(actual, predictions, labels=CLASSES, average="macro")),
        "log_loss": float(log_loss(actual, probabilities, labels=CLASSES)),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: acurácia={metrics['accuracy']:.3f} "
        f"F1-macro={metrics['f1_macro']:.3f} "
        f"log loss={metrics['log_loss']:.3f}"
    )


def prediction_table(model, matches: pd.DataFrame) -> pd.DataFrame:
    """Relaciona cada previsão e probabilidade à partida correspondente."""
    features = matches.loc[:, FEATURE_COLUMNS]
    probabilities = pd.DataFrame(
        model.predict_proba(features),
        columns=[f"prob_{label}" for label in model.classes_],
        index=matches.index,
    )
    predictions = matches[["id", "data", "mandante", "visitante", "resultado"]].copy()
    predictions["previsto"] = model.predict(features)
    return predictions.join(probabilities)


def validation_diagnostics(predictions: pd.DataFrame) -> dict:
    """Descreve erros de 2023 sem usar esse conjunto para treinar o modelo."""
    actual = predictions["resultado"]
    predicted = predictions["previsto"]
    matrix = confusion_matrix(actual, predicted, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0
    )
    by_class = {
        label: {
            "actual": int(support[index]),
            "predicted": int((predicted == label).sum()),
            "correct": int(matrix[index, index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(CLASSES)
    }

    confidence = predictions[[f"prob_{label}" for label in CLASSES]].max(axis=1)
    correct = actual.eq(predicted)
    bands = []
    edges = (0.0, 0.4, 0.5, 0.6, 0.7, 1.0)
    for low, high in zip(edges, edges[1:]):
        selected = confidence.ge(low) & (
            confidence.le(high) if high == 1.0 else confidence.lt(high)
        )
        count = int(selected.sum())
        bands.append(
            {
                "from": low,
                "to": high,
                "matches": count,
                "mean_confidence": float(confidence[selected].mean()) if count else None,
                "accuracy": float(correct[selected].mean()) if count else None,
            }
        )

    mistakes = predictions.loc[~correct].copy()
    mistakes["confidence"] = confidence[~correct]
    mistakes = mistakes.sort_values(["confidence", "id"], ascending=[False, True]).head(5)
    examples = [
        {
            "id": int(row.id),
            "date": str(row.data),
            "home": row.mandante,
            "away": row.visitante,
            "actual": row.resultado,
            "predicted": row.previsto,
            "confidence": float(row.confidence),
        }
        for row in mistakes.itertuples()
    ]
    return {
        "season": 2023,
        "training_seasons": [2020, 2021, 2022],
        "classes": list(CLASSES),
        "confusion_matrix": matrix.tolist(),
        "by_class": by_class,
        "confidence_bands": bands,
        "most_confident_errors": examples,
    }


def main() -> None:
    matches = load_feature_table()
    train, validation, test = split_by_season(matches)
    print(f"Treino: 2020-2022 ({len(train)} jogos)")
    print(f"Validação: 2023 ({len(validation)} jogos)")
    print(f"Teste final: 2024 ({len(test)} jogos)")

    # A validação compara métodos sem consultar os resultados de 2024.
    model = make_model().fit(train.loc[:, FEATURE_COLUMNS], train["resultado"])
    reference = DummyClassifier(strategy="prior").fit(
        train.loc[:, FEATURE_COLUMNS], train["resultado"]
    )
    validation_metrics = {
        "reference": evaluate(reference, validation),
        "logistic_regression": evaluate(model, validation),
    }
    print_metrics("Validação 2023 | Referência", validation_metrics["reference"])
    print_metrics("Validação 2023 | Regressão logística", validation_metrics["logistic_regression"])

    validation_predictions = prediction_table(model, validation)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    diagnostics = validation_diagnostics(validation_predictions)
    draws = diagnostics["by_class"]["D"]
    print(
        "Validação 2023 | Empates: "
        f"{draws['correct']}/{draws['actual']} reconhecidos; "
        f"{draws['predicted']} previsões de empate."
    )

    # Com o método fixado, 2023 pode entrar no treino antes do teste final.
    development = pd.concat([train, validation], ignore_index=True)
    final_model = make_model().fit(
        development.loc[:, FEATURE_COLUMNS], development["resultado"]
    )
    final_reference = DummyClassifier(strategy="prior").fit(
        development.loc[:, FEATURE_COLUMNS], development["resultado"]
    )
    test_metrics = {
        "reference": evaluate(final_reference, test),
        "logistic_regression": evaluate(final_model, test),
    }
    print_metrics("Teste 2024 | Referência", test_metrics["reference"])
    print_metrics("Teste 2024 | Regressão logística", test_metrics["logistic_regression"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    report = {
        "target": "resultado",
        "classes": list(CLASSES),
        "features": list(FEATURE_COLUMNS),
        "split": {"train": [2020, 2021, 2022], "validation": [2023], "test": [2024]},
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation": validation_metrics,
        "test": test_metrics,
    }
    METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    VALIDATION_DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n"
    )

    predictions = prediction_table(final_model, test)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Diagnóstico de 2023 salvo em: {VALIDATION_DIAGNOSTICS_PATH}")
    print(f"Previsões de validação salvas em: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Previsões retrospectivas salvas em: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
