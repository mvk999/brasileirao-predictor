"""Compara janelas de treino e pesos por recência em temporadas completas.

Uso: python -m src.model.experiment_long_history
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data.build_features import build_features
from src.data.prepare_matches import load_raw_matches, prepare_matches
from src.model.train_baseline import ARTIFACTS_DIR, CLASSES, FEATURE_COLUMNS, PROJECT_ROOT, make_model


YEARS = (*range(2006, 2016), *range(2017, 2025))
BACKTEST_YEARS = (*range(2018, 2023),)
EVALUATION_YEARS = (2023, 2024)
STRATEGIES = (
    "recent_3", "recent_5", "recent_10", "all",
    "weighted_hl_1", "weighted_hl_2", "weighted_hl_4", "weighted_hl_8",
)
REPORT_PATH = ARTIFACTS_DIR / "long_history_experiment.json"
FEATURES_PATH = PROJECT_ROOT / "data/processed/matches_features_2006_2024.csv"
CURRENT_FEATURES_PATH = PROJECT_ROOT / "data/processed/matches_features_2020_2024.csv"


def load_complete_history() -> pd.DataFrame:
    """Usa somente anos de 20 clubes com 380 jogos completos na fonte local."""
    raw = load_raw_matches()
    matches = prepare_matches(raw, seasons=YEARS)
    counts = matches["temporada"].value_counts().sort_index().to_dict()
    if counts != {year: 380 for year in YEARS}:
        raise ValueError(f"Temporadas completas esperadas: {YEARS}; encontrado {counts}.")
    for year, season in matches.groupby("temporada"):
        teams = set(season["mandante"]) | set(season["visitante"])
        if len(teams) != 20 or season["rodada"].nunique() != 38 or not season.groupby("rodada").size().eq(10).all():
            raise ValueError(f"Temporada {year} não tem 20 clubes e 38 rodadas de dez partidas.")
    return matches


def compare_existing_features(features: pd.DataFrame) -> None:
    """Confere que o gerador Python reproduz os 12 atributos do notebook."""
    if not CURRENT_FEATURES_PATH.is_file():
        return
    current = pd.read_csv(CURRENT_FEATURES_PATH).set_index("id").sort_index()
    generated = features[features["temporada"].between(2020, 2024)].set_index("id").sort_index()
    if not current.index.equals(generated.index):
        raise ValueError("IDs de 2020–2024 diferem da base do notebook.")
    for name in FEATURE_COLUMNS:
        if not np.allclose(current[name], generated[name], equal_nan=True, rtol=0, atol=1e-12):
            raise ValueError(f"Atributo {name} difere do notebook.")


def recency_weights(training_seasons: pd.Series, prediction_year: int, half_life: int) -> np.ndarray:
    """Peso cai à metade a cada `half_life` anos; média um mantém a escala do ajuste."""
    if half_life <= 0 or training_seasons.empty or training_seasons.ge(prediction_year).any():
        raise ValueError("Pesos exigem meia-vida positiva e somente temporadas anteriores.")
    age = prediction_year - 1 - training_seasons.to_numpy(dtype=int)
    weights = np.power(0.5, age / half_life)
    return weights / weights.mean()


def evaluate_year(features: pd.DataFrame, year: int, strategy: str) -> dict:
    previous = sorted(set(features["temporada"]) & set(range(year)))
    training_years = (
        previous[-int(strategy.removeprefix("recent_")):]
        if strategy.startswith("recent_") else previous
    )
    train = features[features["temporada"].isin(training_years)]
    validation = features[features["temporada"] == year]
    if len(validation) != 380 or len(train) != len(training_years) * 380:
        raise ValueError(f"Dados incompletos ao avaliar {year} com {strategy}.")
    fit_options = {}
    if strategy.startswith("weighted_hl_"):
        half_life = int(strategy.removeprefix("weighted_hl_"))
        fit_options["logisticregression__sample_weight"] = recency_weights(
            train["temporada"], year, half_life
        )
    elif strategy != "all" and not strategy.startswith("recent_"):
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    model = make_model().fit(train.loc[:, FEATURE_COLUMNS], train["resultado"], **fit_options)
    predictions = model.predict(validation.loc[:, FEATURE_COLUMNS])
    probabilities = model.predict_proba(validation.loc[:, FEATURE_COLUMNS])
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    precision, recall, f1, support = precision_recall_fscore_support(
        validation["resultado"], predictions, labels=CLASSES, zero_division=0
    )
    return {
        "training_years": training_years,
        "training_games": len(train),
        "weighting": strategy if strategy.startswith("weighted_hl_") else "uniform",
        "accuracy": float(accuracy_score(validation["resultado"], predictions)),
        "f1_macro": float(f1_score(validation["resultado"], predictions, labels=CLASSES, average="macro")),
        "log_loss": float(log_loss(validation["resultado"], probabilities, labels=CLASSES)),
        "by_class": {
            label: {
                "actual": int(support[index]),
                "predicted": int((predictions == label).sum()),
                "correct": int(((validation["resultado"].to_numpy() == label) & (predictions == label)).sum()),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(CLASSES)
        },
    }


def main() -> None:
    matches = load_complete_history()
    features = build_features(matches)
    compare_existing_features(features)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    report = {
        "source_years": list(YEARS),
        "source_games": len(matches),
        "excluded_years": {
            "2000-2002": "fonte não contém esses anos",
            "2003-2005": "formato de 24/22 clubes, não comparável diretamente ao recorte de 20 clubes",
            "2016": "fonte local contém 379, não 380 jogos",
        },
        "feature_count": len(FEATURE_COLUMNS),
        "feature_semantics": "últimos cinco jogos do clube dentro da mesma temporada, sem resultado atual",
        "weighting_semantics": "meia-vida em anos; peso do ano anterior igual a 1 antes de normalizar; pesos médios iguais a 1; apenas regressão recebe sample_weight",
        "backtest": {},
        "retrospective": {},
        "strategies": list(STRATEGIES),
    }
    for year in (*BACKTEST_YEARS, *EVALUATION_YEARS):
        section = "backtest" if year in BACKTEST_YEARS else "retrospective"
        report[section][str(year)] = {
            strategy: evaluate_year(features, year, strategy) for strategy in STRATEGIES
        }
        scores = report[section][str(year)]
        print(
            f"{year}: " + " | ".join(
                f"{strategy} acurácia={scores[strategy]['accuracy']:.3f} "
                f"F1={scores[strategy]['f1_macro']:.3f}"
                for strategy in STRATEGIES
            )
        )
    for strategy in STRATEGIES:
        rows = [report["backtest"][str(year)][strategy] for year in BACKTEST_YEARS]
        report.setdefault("backtest_average", {})[strategy] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("accuracy", "f1_macro", "log_loss")
        }
    report["best_backtest_macro_f1"] = max(
        STRATEGIES, key=lambda strategy: report["backtest_average"][strategy]["f1_macro"]
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Partidas: {len(matches)} | Temporadas completas: {len(YEARS)} | Features: {len(FEATURE_COLUMNS)}")
    print(f"Melhor F1 macro médio em 2018–2022: {report['best_backtest_macro_f1']}")
    print(f"Relatório: {REPORT_PATH}; base: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
