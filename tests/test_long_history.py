"""Confere atributos históricos e o recorte configurável de temporadas."""

import pandas as pd

from src.data.build_features import build_features
from src.data.prepare_matches import prepare_matches


def test_features_use_previous_games_and_reset_at_new_season() -> None:
    games = pd.DataFrame(
        {
            "id": [3, 1, 2],
            "temporada": [2024, 2023, 2023],
            "rodada": [1, 1, 2],
            "data": ["2024-04-01", "2023-04-01", "2023-04-08"],
            "mandante": ["Casa", "Casa", "Fora"],
            "visitante": ["Outro", "Fora", "Casa"],
            "gols_mandante": [5, 2, 0],
            "gols_visitante": [0, 1, 1],
            "resultado": ["H", "H", "A"],
        }
    )

    generated = build_features(games).set_index("id")
    changed_current = build_features(games.assign(gols_mandante=[5, 2, 9])).set_index("id")

    assert generated.loc[2, "visitante_pontos_por_jogo_ultimos_5"] == 3
    assert generated.loc[2, "visitante_media_gols_pro_ultimos_5"] == 2
    assert pd.isna(generated.loc[2, "visitante_pontos_por_jogo_mesmo_mando_ultimos_5"])
    assert pd.isna(generated.loc[3, "mandante_pontos_por_jogo_ultimos_5"])
    assert generated.loc[2, "visitante_media_gols_pro_ultimos_5"] == changed_current.loc[2, "visitante_media_gols_pro_ultimos_5"]


def test_prepare_matches_accepts_explicit_older_season() -> None:
    raw = pd.DataFrame(
        {
            "ID": [1], "rodata": [1], "data": ["01/05/2019"],
            "mandante": ["Casa"], "visitante": ["Fora"],
            "mandante_Placar": [1], "visitante_Placar": [0],
            "arena": ["A"], "mandante_Estado": ["SP"], "visitante_Estado": ["RJ"],
        }
    )

    assert prepare_matches(raw).empty
    older = prepare_matches(raw, seasons=(2019,))
    assert len(older) == 1
    assert older.loc[0, "temporada"] == 2019
