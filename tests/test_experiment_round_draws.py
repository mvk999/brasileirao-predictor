"""Confere o teto de empates e o instantâneo anterior à rodada."""

import numpy as np
import pandas as pd

from src.model.experiment_round_draws import choose_round_labels, pre_round_predictions


def test_round_rule_limits_draws_and_prefers_smallest_goal_gap() -> None:
    predictions = pd.DataFrame(
        {
            "id": range(1, 9),
            "rodada": [1] * 7 + [2],
            "gols_estimados_mandante": [1.0] * 8,
            "gols_estimados_visitante": [1.0 + gap for gap in (.01, .02, .03, .04, .05, .06, .07, .01)],
            "diferenca_gols_estimados": [.01, .02, .03, .04, .05, .06, .07, .01],
            "resultado": ["H"] * 8,
        }
    )

    chosen = choose_round_labels(predictions, gap_limit=.10, max_draws=5)

    assert chosen.tolist() == ["D", "D", "D", "D", "D", "A", "A", "D"]
    assert np.array_equal(chosen, choose_round_labels(predictions.assign(resultado="D"), .10, 5))


def test_all_matches_in_round_use_same_prior_snapshot() -> None:
    prior = {
        "id": 0, "temporada": 2021, "rodada": 1, "data": pd.Timestamp("2021-01-01"),
        "mandante": "Casa", "visitante": "Fora", "gols_mandante": 2,
        "gols_visitante": 1, "resultado": "H",
    }
    matches = [prior]
    for round_number in range(1, 39):
        start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=(round_number - 1) * 7)
        for offset in range(10):
            matches.append(
                {
                    **prior,
                    "id": (round_number - 1) * 10 + offset + 1,
                    "temporada": 2023,
                    "rodada": round_number,
                    "data": start + pd.Timedelta(days=int(offset >= 5)),
                    "gols_mandante": 0 if offset >= 5 else 3,
                    "gols_visitante": 3 if offset >= 5 else 0,
                }
            )

    predictions = pre_round_predictions(pd.DataFrame(matches), 2023, prior_games=5)
    first_round = predictions[predictions["rodada"] == 1]
    assert first_round["gols_estimados_mandante"].nunique() == 1
    assert first_round["gols_estimados_visitante"].nunique() == 1
