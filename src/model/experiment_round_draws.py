"""Testa empates por proximidade das taxas de gols, com teto por rodada.

Uso: python -m src.model.experiment_round_draws
O limite é escolhido em 2022; 2023 é usado para comparação retrospectiva.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.model.experiment_draws import score
from src.model.experiment_poisson import (
    expected_goals,
    historical_strengths,
    load_matches,
    result_probabilities,
)
from src.model.train_baseline import ARTIFACTS_DIR, CLASSES


REPORT_PATH = ARTIFACTS_DIR / "round_draw_experiment.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "round_draw_predictions_2023.csv"
MAX_DRAWS_PER_ROUND = 5
PRIOR_GAMES = 5  # Escolhido na validação de 2022 do experimento Poisson.
GAP_LIMITS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)


def pre_round_predictions(matches: pd.DataFrame, season: int, prior_games: int) -> pd.DataFrame:
    """Todos os jogos da rodada usam o histórico anterior ao primeiro jogo dela."""
    evaluation = matches[matches["temporada"] == season]
    if len(evaluation) != 380 or evaluation["rodada"].nunique() != 38:
        raise ValueError(f"Temporada {season} precisa de 38 rodadas e 380 jogos.")
    rows = []
    for round_number, games in evaluation.groupby("rodada", sort=True):
        round_start = games["data"].min()
        history = matches[(matches["data"] < round_start) & (matches["temporada"] <= season)]
        context = historical_strengths(history, season, prior_games)
        for game in games.itertuples(index=False):
            home_rate, away_rate = expected_goals(game.mandante, game.visitante, context)
            away_prob, draw_prob, home_prob = result_probabilities(home_rate, away_rate)
            rows.append(
                {
                    "id": game.id,
                    "temporada": season,
                    "rodada": round_number,
                    "data": game.data.strftime("%Y-%m-%d"),
                    "mandante": game.mandante,
                    "visitante": game.visitante,
                    "resultado": game.resultado,
                    "gols_estimados_mandante": home_rate,
                    "gols_estimados_visitante": away_rate,
                    "diferenca_gols_estimados": abs(home_rate - away_rate),
                    "prob_A": away_prob,
                    "prob_D": draw_prob,
                    "prob_H": home_prob,
                }
            )
    return pd.DataFrame(rows).sort_values(["rodada", "id"]).reset_index(drop=True)


def choose_round_labels(predictions: pd.DataFrame, gap_limit: float, max_draws: int = MAX_DRAWS_PER_ROUND) -> np.ndarray:
    """Seleciona até max_draws jogos elegíveis, por menor diferença de gols."""
    if gap_limit < 0 or max_draws < 0:
        raise ValueError("Limite e teto precisam ser não negativos.")
    labels = pd.Series(
        np.where(
            predictions["gols_estimados_mandante"].to_numpy()
            >= predictions["gols_estimados_visitante"].to_numpy(),
            "H", "A",
        ),
        index=predictions.index,
    )
    for _, games in predictions.groupby("rodada", sort=True):
        eligible = games[games["diferenca_gols_estimados"] <= gap_limit]
        selected = eligible.sort_values(["diferenca_gols_estimados", "id"]).head(max_draws)
        labels.loc[selected.index] = "D"
    return labels.to_numpy()


def evaluate(predictions: pd.DataFrame, labels: np.ndarray) -> dict:
    probabilities = predictions[[f"prob_{label}" for label in CLASSES]].to_numpy()
    metrics = score(predictions["resultado"], labels, probabilities)
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["resultado"], labels, labels=CLASSES, zero_division=0
    )
    metrics["by_class"] = {
        label: {
            "actual": int(support[index]),
            "predicted": int((labels == label).sum()),
            "correct": int(((predictions["resultado"].to_numpy() == label) & (labels == label)).sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(CLASSES)
    }
    per_round = pd.Series(labels == "D", index=predictions.index).groupby(predictions["rodada"]).sum()
    metrics["max_predicted_draws_in_round"] = int(per_round.max())
    return metrics


def main() -> None:
    matches = load_matches()
    prior_games = PRIOR_GAMES
    selection = pre_round_predictions(matches, 2022, prior_games)
    validation = pre_round_predictions(matches, 2023, prior_games)
    selection_scores = {
        f"{limit:.2f}": evaluate(selection, choose_round_labels(selection, limit))
        for limit in GAP_LIMITS
    }
    chosen = max(
        GAP_LIMITS,
        key=lambda limit: (selection_scores[f"{limit:.2f}"]["f1_macro"], -limit),
    )
    original_labels = np.asarray(CLASSES)[
        np.argmax(validation[[f"prob_{label}" for label in CLASSES]].to_numpy(), axis=1)
    ]
    heuristic_labels = choose_round_labels(validation, chosen)
    validation["previsto_poisson"] = original_labels
    validation["previsto_regra_rodada"] = heuristic_labels
    round_draws = validation["resultado"].eq("D").groupby(validation["rodada"]).sum()
    report = {
        "model": "independent Poisson with pre-round historical snapshot",
        "prior_games": prior_games,
        "max_draws_per_round": MAX_DRAWS_PER_ROUND,
        "gap_limit_candidates": list(GAP_LIMITS),
        "selection_objective": "maximize macro F1 in 2022, tie prefers smaller goal-gap limit",
        "chosen_gap_limit": chosen,
        "selection_2022": selection_scores,
        "validation_2023": {
            "poisson_argmax": evaluate(validation, original_labels),
            "round_rule": evaluate(validation, heuristic_labels),
        },
        "actual_draws_per_round_2023": {
            "maximum": int(round_draws.max()),
            "rounds_above_cap": int((round_draws > MAX_DRAWS_PER_ROUND).sum()),
            "round_numbers_above_cap": [int(value) for value in round_draws[round_draws > MAX_DRAWS_PER_ROUND].index],
        },
        "equal_estimated_goals_2023": int(validation["diferenca_gols_estimados"].eq(0).sum()),
        "test_2024_used": False,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation.to_csv(PREDICTIONS_PATH, index=False)
    for name, item in report["validation_2023"].items():
        print(
            f"{name}: acurácia={item['accuracy']:.3f} F1-macro={item['f1_macro']:.3f} "
            f"D={item['by_class']['D']['correct']}/{item['by_class']['D']['predicted']} "
            f"H={item['by_class']['H']['correct']}/{item['by_class']['H']['actual']} "
            f"A={item['by_class']['A']['correct']}/{item['by_class']['A']['actual']}"
        )
    print(f"Limite escolhido em 2022: {chosen:.2f} gol; teto: {MAX_DRAWS_PER_ROUND} empates/rodada")
    print(f"Relatório: {REPORT_PATH}; previsões: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
