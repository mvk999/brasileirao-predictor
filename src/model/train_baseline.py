"""Treina uma primeira referência para os resultados H, D e A."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data/processed/matches_features_2020_2024.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "logistic_baseline_2020_2023.joblib"
METRICS_PATH = ARTIFACTS_DIR / "logistic_baseline_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/predictions_2024.csv"

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

    probabilities = pd.DataFrame(
        final_model.predict_proba(test.loc[:, FEATURE_COLUMNS]),
        columns=[f"prob_{label}" for label in final_model.classes_],
        index=test.index,
    )
    predictions = test[["id", "data", "mandante", "visitante", "resultado"]].copy()
    predictions["previsto"] = final_model.predict(test.loc[:, FEATURE_COLUMNS])
    predictions = predictions.join(probabilities)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Previsões retrospectivas salvas em: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
