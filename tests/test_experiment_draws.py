"""Confere a regra experimental de empate e o uso apenas de médias históricas."""

import numpy as np
import pandas as pd

from src.model.experiment_draws import add_balance_features, choose_label


def test_balance_features_ignore_current_score_and_result() -> None:
    historical = pd.DataFrame(
        {
            "mandante_pontos_por_jogo_ultimos_5": [2.0],
            "visitante_pontos_por_jogo_ultimos_5": [1.0],
            "mandante_media_gols_pro_ultimos_5": [2.0],
            "visitante_media_gols_contra_ultimos_5": [1.0],
            "visitante_media_gols_pro_ultimos_5": [1.0],
            "mandante_media_gols_contra_ultimos_5": [1.0],
            "gols_mandante": [4],
            "gols_visitante": [0],
            "resultado": ["H"],
        }
    )
    changed_score = historical.assign(gols_mandante=0, gols_visitante=4, resultado="A")

    original = add_balance_features(historical)
    changed = add_balance_features(changed_score)

    for column in ("diferenca_absoluta_pontos", "diferenca_absoluta_gols_esperados", "interacao_equilibrio_gols"):
        assert original.loc[0, column] == changed.loc[0, column]
    assert original.loc[0, "diferenca_absoluta_pontos"] == 1.0
    assert original.loc[0, "diferenca_absoluta_gols_esperados"] == 0.5
    assert original.loc[0, "interacao_equilibrio_gols"] == 1.25


def test_draw_threshold_falls_back_to_best_non_draw_class() -> None:
    classes = np.array(["A", "D", "H"])
    probabilities = np.array([[0.25, 0.34, 0.41], [0.4, 0.31, 0.29]])

    assert choose_label(probabilities, classes, None).tolist() == ["H", "A"]
    assert choose_label(probabilities, classes, 0.33).tolist() == ["D", "A"]
