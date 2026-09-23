"""Avalia gols via Poisson independente com informações anteriores ao jogo.

Uso, na raiz do projeto: python -m src.model.experiment_poisson
O experimento não altera o modelo principal nem usa resultados de 2024.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.model.experiment_draws import score
from src.model.train_baseline import (
    ARTIFACTS_DIR,
    CLASSES,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    load_feature_table,
    make_model,
)


MATCHES_PATH = PROJECT_ROOT / "data/processed/matches_2020_2024.csv"
REPORT_PATH = ARTIFACTS_DIR / "poisson_experiment.json"
PRIOR_GAMES = (5, 10, 20, 40)


def load_matches(path=MATCHES_PATH) -> pd.DataFrame:
    matches = pd.read_csv(path, parse_dates=["data"])
    required = {"id", "temporada", "data", "mandante", "visitante", "gols_mandante", "gols_visitante", "resultado"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {', '.join(sorted(missing))}")
    if not matches["id"].is_unique or matches[list(required)].isna().any().any():
        raise ValueError("Partidas precisam de IDs únicos e dados completos.")
    if (matches[["gols_mandante", "gols_visitante"]] < 0).any().any():
        raise ValueError("Gols não podem ser negativos.")
    if len(matches) != 1900 or matches["temporada"].value_counts().to_dict() != {year: 380 for year in range(2020, 2025)}:
        raise ValueError("Esperadas 380 partidas por temporada de 2020 a 2024.")
    return matches.sort_values(["data", "id"]).reset_index(drop=True)


def result_probabilities(home_rate: float, away_rate: float) -> np.ndarray:
    """Skellam é a distribuição da diferença entre dois Poissons independentes."""
    if home_rate <= 0 or away_rate <= 0:
        raise ValueError("As taxas de gols devem ser positivas.")
    away = float(skellam.cdf(-1, home_rate, away_rate))
    draw = float(skellam.pmf(0, home_rate, away_rate))
    home = float(skellam.sf(0, home_rate, away_rate))
    return np.array([away, draw, home])


def historical_strengths(history: pd.DataFrame, season: int, prior_games: int):
    """Mede ataque e vulnerabilidade defensiva com encolhimento para a liga."""
    weights = np.power(0.5, season - history["temporada"].to_numpy())
    home_goals = history["gols_mandante"].to_numpy()
    away_goals = history["gols_visitante"].to_numpy()
    league_home = float(np.average(home_goals, weights=weights))
    league_away = float(np.average(away_goals, weights=weights))
    totals = defaultdict(lambda: np.zeros(4))  # ataque obs/esperado; defesa obs/esperado
    for match, weight in zip(history.itertuples(index=False), weights):
        home = totals[match.mandante]
        away = totals[match.visitante]
        home[0] += weight * match.gols_mandante
        home[1] += weight * league_home
        home[2] += weight * match.gols_visitante
        home[3] += weight * league_away
        away[0] += weight * match.gols_visitante
        away[1] += weight * league_away
        away[2] += weight * match.gols_mandante
        away[3] += weight * league_home
    prior_goals = prior_games * (league_home + league_away) / 2
    return league_home, league_away, totals, prior_goals


def expected_goals(home_team: str, away_team: str, context) -> tuple[float, float]:
    league_home, league_away, totals, prior_goals = context
    home = totals[home_team]
    away = totals[away_team]
    attack_home = (home[0] + prior_goals) / (home[1] + prior_goals)
    defense_home = (home[2] + prior_goals) / (home[3] + prior_goals)
    attack_away = (away[0] + prior_goals) / (away[1] + prior_goals)
    defense_away = (away[2] + prior_goals) / (away[3] + prior_goals)
    return league_home * attack_home * defense_away, league_away * attack_away * defense_home


def predict_season(matches: pd.DataFrame, season: int, prior_games: int) -> tuple[np.ndarray, pd.DataFrame]:
    evaluation = matches[matches["temporada"] == season].copy()
    if len(evaluation) != 380:
        raise ValueError(f"Temporada {season} precisa de 380 jogos.")
    predictions = {}
    for date, daily_matches in evaluation.groupby("data", sort=True):
        history = matches[(matches["data"] < date) & (matches["temporada"] <= season)]
        if history.empty:
            raise ValueError(f"Não há histórico antes de {date}.")
        context = historical_strengths(history, season, prior_games)
        for match in daily_matches.itertuples(index=False):
            predictions[match.id] = result_probabilities(*expected_goals(match.mandante, match.visitante, context))
    probabilities = np.stack([predictions[match_id] for match_id in evaluation["id"]])
    return probabilities, evaluation


def summarize(actual: pd.Series, probabilities: np.ndarray) -> dict:
    predicted = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    metrics = score(actual, predicted, probabilities)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0
    )
    metrics["by_class"] = {
        label: {
            "actual": int(support[index]),
            "predicted": int((predicted == label).sum()),
            "correct": int(((actual.to_numpy() == label) & (predicted == label)).sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(CLASSES)
    }
    metrics["confusion_matrix"] = confusion_matrix(actual, predicted, labels=CLASSES).tolist()
    metrics["mean_draw_probability"] = float(probabilities[:, 1].mean())
    return metrics


def main() -> None:
    matches = load_matches()
    selection = {}
    for prior in PRIOR_GAMES:
        probabilities, games = predict_season(matches, 2022, prior)
        selection[prior] = summarize(games["resultado"], probabilities)
    chosen = min(PRIOR_GAMES, key=lambda prior: (selection[prior]["log_loss"], prior))
    poisson_prob, validation = predict_season(matches, 2023, chosen)

    features = load_feature_table()
    train = features[features["temporada"].isin((2020, 2021, 2022))]
    logistic_validation = features[features["temporada"] == 2023]
    logistic = make_model().fit(train.loc[:, FEATURE_COLUMNS], train["resultado"])
    logistic_prob = logistic.predict_proba(logistic_validation.loc[:, FEATURE_COLUMNS])
    logistic_prob = logistic_prob[:, [list(logistic.classes_).index(label) for label in CLASSES]]

    report = {
        "method": "independent Poisson with season-decayed team attack and defensive weakness",
        "selection": "prior_games selected on 2022 by lowest 3-class log loss",
        "selection_2022": {str(prior): item for prior, item in selection.items()},
        "chosen_prior_games": chosen,
        "validation_2023": {
            "logistic_baseline": summarize(logistic_validation["resultado"], logistic_prob),
            "poisson": summarize(validation["resultado"], poisson_prob),
        },
        "test_2024_used": False,
        "dixon_coles_correction_used": False,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, item in report["validation_2023"].items():
        print(
            f"{name}: acurácia={item['accuracy']:.3f} F1-macro={item['f1_macro']:.3f} "
            f"log loss={item['log_loss']:.3f} | D={item['by_class']['D']['correct']}/"
            f"{item['by_class']['D']['predicted']} H={item['by_class']['H']['correct']}/"
            f"{item['by_class']['H']['actual']} A={item['by_class']['A']['correct']}/"
            f"{item['by_class']['A']['actual']}"
        )
    print(f"Prior escolhido em 2022: {chosen} jogos; relatório: {REPORT_PATH}")


if __name__ == "__main__":
    main()
