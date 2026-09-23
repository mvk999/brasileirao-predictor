"""Verifica probabilidades e forças do experimento de gols."""

import numpy as np
import pandas as pd

from src.model.experiment_poisson import expected_goals, historical_strengths, result_probabilities


def test_equal_goal_rates_have_symmetric_win_probabilities() -> None:
    away, draw, home = result_probabilities(1.2, 1.2)

    assert np.isclose(away + draw + home, 1.0)
    assert np.isclose(away, home)
    assert 0 < draw < 1


def test_unseen_teams_return_league_goal_rates() -> None:
    history = pd.DataFrame(
        {
            "temporada": [2021, 2021],
            "mandante": ["A", "C"],
            "visitante": ["B", "D"],
            "gols_mandante": [2, 1],
            "gols_visitante": [1, 0],
        }
    )
    context = historical_strengths(history, 2022, prior_games=10)

    assert expected_goals("Novo casa", "Novo fora", context) == (1.5, 0.5)
