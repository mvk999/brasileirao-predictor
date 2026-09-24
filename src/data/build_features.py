"""Gera os 12 atributos históricos para temporadas completas arbitrárias."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = (
    "pontos_por_jogo_ultimos_5",
    "media_gols_pro_ultimos_5",
    "media_gols_contra_ultimos_5",
    "pontos_por_jogo_mesmo_mando_ultimos_5",
    "media_gols_pro_mesmo_mando_ultimos_5",
    "media_gols_contra_mesmo_mando_ultimos_5",
)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Cada média usa até cinco jogos anteriores do clube na mesma temporada."""
    required = {"id", "temporada", "rodada", "data", "mandante", "visitante", "resultado", "gols_mandante", "gols_visitante"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para gerar atributos: {', '.join(sorted(missing))}")
    if not matches["id"].is_unique:
        raise ValueError("IDs de partidas precisam ser únicos.")
    games = matches.copy()
    games["data"] = pd.to_datetime(games["data"])
    home = pd.DataFrame(
        {
            "id": games["id"], "temporada": games["temporada"], "data": games["data"],
            "clube": games["mandante"], "mando": "casa",
            "pontos": np.select([games["resultado"].eq("H"), games["resultado"].eq("D")], [3, 1], default=0),
            "gols_pro": games["gols_mandante"], "gols_contra": games["gols_visitante"],
        }
    )
    away = pd.DataFrame(
        {
            "id": games["id"], "temporada": games["temporada"], "data": games["data"],
            "clube": games["visitante"], "mando": "fora",
            "pontos": np.select([games["resultado"].eq("A"), games["resultado"].eq("D")], [3, 1], default=0),
            "gols_pro": games["gols_visitante"], "gols_contra": games["gols_mandante"],
        }
    )
    history = pd.concat([home, away], ignore_index=True)
    history = history.sort_values(["temporada", "data", "id", "mando"]).reset_index(drop=True)
    for suffix, groups in (("ultimos_5", ["temporada", "clube"]),
                           ("mesmo_mando_ultimos_5", ["temporada", "clube", "mando"])):
        for source, target in (("pontos", "pontos_por_jogo"),
                               ("gols_pro", "media_gols_pro"),
                               ("gols_contra", "media_gols_contra")):
            history[f"{target}_{suffix}"] = history.groupby(groups)[source].transform(
                lambda values: values.shift(1).rolling(5, min_periods=1).mean()
            )
    home_features = history.loc[history["mando"] == "casa", ["id", *BASE_FEATURES]].rename(
        columns={name: f"mandante_{name}" for name in BASE_FEATURES}
    )
    away_features = history.loc[history["mando"] == "fora", ["id", *BASE_FEATURES]].rename(
        columns={name: f"visitante_{name}" for name in BASE_FEATURES}
    )
    result = (
        games[["id", "temporada", "rodada", "data", "mandante", "visitante", "resultado"]]
        .merge(home_features, on="id", validate="one_to_one")
        .merge(away_features, on="id", validate="one_to_one")
        .sort_values(["temporada", "data", "id"])
        .reset_index(drop=True)
    )
    if len(result) != len(matches):
        raise ValueError("A geração de atributos alterou a quantidade de jogos.")
    return result
