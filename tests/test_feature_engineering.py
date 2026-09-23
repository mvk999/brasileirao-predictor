"""Protege a ordem temporal das janelas calculadas no notebook."""

import json
from pathlib import Path

import pandas as pd


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks/02_feature_engineering.ipynb"


def test_recent_form_uses_home_and_away_matches_in_date_order() -> None:
    matches = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "temporada": [2024] * 4,
            "rodada": [1, 2, 3, 4],
            "data": pd.to_datetime(
                ["2024-04-01", "2024-04-08", "2024-04-15", "2024-04-22"]
            ),
            "mandante": ["A", "C", "A", "B"],
            "visitante": ["B", "A", "C", "A"],
            "gols_mandante": [2, 1, 0, 1],
            "gols_visitante": [0, 1, 2, 0],
            "resultado": ["H", "D", "A", "H"],
        }
    )

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"pd": pd, "df": matches}
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    start = next(i for i, source in enumerate(sources) if source.startswith("jogos_mandante = df["))
    end = next(
        i for i, source in enumerate(sources)
        if source.startswith('historico_clubes["media_saldo_mesmo_mando_ultimos_5"]')
    )

    for source in sources[start : end + 1]:
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    history = namespace["historico_clubes"]
    club_a = history[history["clube"] == "A"].reset_index(drop=True)

    assert club_a["id"].tolist() == [1, 2, 3, 4]
    assert pd.isna(club_a.loc[0, "pontos_por_jogo_ultimos_5"])
    assert club_a["pontos_ultimos_5"].tolist()[1:] == [3, 4, 4]
    assert club_a.loc[2, "pontos_por_jogo_ultimos_5"] == 2
    assert club_a.loc[2, "pontos_mesmo_mando_ultimos_5"] == 3
    assert club_a.loc[3, "pontos_mesmo_mando_ultimos_5"] == 1
