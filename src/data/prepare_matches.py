"""Prepara as partidas do Campeonato Brasileiro para uso no projeto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_MATCHES_PATH = PROJECT_ROOT / "data/raw/campeonato-brasileiro-full.csv"
PROCESSED_MATCHES_PATH = PROJECT_ROOT / "data/processed/matches_2020_2024.csv"
SEASONS = (2020, 2021, 2022, 2023, 2024)
REQUIRED_RAW_COLUMNS = {
    "ID",
    "rodata",
    "data",
    "mandante",
    "visitante",
    "mandante_Placar",
    "visitante_Placar",
    "arena",
    "mandante_Estado",
    "visitante_Estado",
}
ESSENTIAL_COLUMNS = (
    "temporada",
    "rodada",
    "data",
    "mandante",
    "visitante",
    "gols_mandante",
    "gols_visitante",
    "resultado",
)


def load_raw_matches(path: Path = RAW_MATCHES_PATH) -> pd.DataFrame:
    """Carrega o CSV bruto e verifica se as colunas necessárias existem."""
    if not path.is_file():
        raise FileNotFoundError(
            "Dataset bruto não encontrado. Esperado em: " f"{path}"
        )

    matches = pd.read_csv(path)
    missing_columns = REQUIRED_RAW_COLUMNS - set(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"O dataset bruto não possui as colunas necessárias: {missing}")
    return matches


def prepare_matches(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Aplica as regras de limpeza e padronização definidas na análise."""
    matches = raw_matches.copy()
    matches["data"] = pd.to_datetime(matches["data"], format="%d/%m/%Y")
    matches["temporada"] = matches["data"].dt.year

    # O Brasileirão 2020 foi concluído em janeiro/fevereiro de 2021 devido à
    # pandemia; a temporada 2021 começou somente em 29/05/2021.
    season_2020_completion = matches["data"].between("2021-01-01", "2021-02-25")
    matches.loc[season_2020_completion, "temporada"] = 2020

    matches = matches[matches["temporada"].isin(SEASONS)].copy()
    matches = matches[
        [
            "ID",
            "temporada",
            "rodata",
            "data",
            "mandante",
            "visitante",
            "mandante_Placar",
            "visitante_Placar",
            "arena",
            "mandante_Estado",
            "visitante_Estado",
        ]
    ].rename(
        columns={
            "ID": "id",
            "rodata": "rodada",
            "mandante_Placar": "gols_mandante",
            "visitante_Placar": "gols_visitante",
            "mandante_Estado": "estado_mandante",
            "visitante_Estado": "estado_visitante",
        }
    )

    matches["resultado"] = "D"
    matches.loc[matches["gols_mandante"] > matches["gols_visitante"], "resultado"] = "H"
    matches.loc[matches["gols_mandante"] < matches["gols_visitante"], "resultado"] = "A"

    return matches.sort_values(["data", "id"]).reset_index(drop=True)


def validate_matches(matches: pd.DataFrame) -> None:
    """Verifica as propriedades esperadas do recorte 2020--2024."""
    if len(matches) != 1900:
        raise ValueError(f"Quantidade de partidas inválida: esperadas 1900, encontradas {len(matches)}.")

    expected_season_counts = {season: 380 for season in SEASONS}
    season_counts = matches["temporada"].value_counts().sort_index().to_dict()
    if season_counts != expected_season_counts:
        raise ValueError(
            "Quantidade de partidas por temporada inválida: "
            f"esperada {expected_season_counts}, encontrada {season_counts}."
        )

    missing_essential = matches[list(ESSENTIAL_COLUMNS)].isna().sum()
    missing_essential = missing_essential[missing_essential > 0]
    if not missing_essential.empty:
        details = ", ".join(f"{column}={count}" for column, count in missing_essential.items())
        raise ValueError(f"Há valores ausentes em colunas essenciais: {details}.")

    invalid_results = set(matches["resultado"].unique()) - {"H", "D", "A"}
    if invalid_results:
        raise ValueError(f"Resultados inválidos encontrados: {sorted(invalid_results)}.")

    duplicate_count = matches.duplicated(
        subset=["temporada", "rodada", "mandante", "visitante"]
    ).sum()
    if duplicate_count:
        raise ValueError(f"Foram encontradas {duplicate_count} duplicatas lógicas.")

    for season, season_matches in matches.groupby("temporada"):
        clubs = set(season_matches["mandante"]) | set(season_matches["visitante"])
        if len(clubs) != 20:
            raise ValueError(f"Temporada {season}: esperados 20 clubes, encontrados {len(clubs)}.")

        home_counts = season_matches["mandante"].value_counts()
        away_counts = season_matches["visitante"].value_counts()
        if not (home_counts.eq(19).all() and away_counts.eq(19).all()):
            raise ValueError(
                f"Temporada {season}: cada clube deve ter 19 jogos como mandante e 19 como visitante."
            )

        round_counts = season_matches.groupby("rodada").size()
        if len(round_counts) != 38 or not round_counts.eq(10).all():
            raise ValueError(
                f"Temporada {season}: esperadas 38 rodadas com 10 partidas cada."
            )


def save_processed_matches(matches: pd.DataFrame, path: Path = PROCESSED_MATCHES_PATH) -> None:
    """Cria o diretório de destino, se preciso, e salva o dataset processado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)


def main() -> None:
    raw_matches = load_raw_matches()
    matches = prepare_matches(raw_matches)
    validate_matches(matches)
    save_processed_matches(matches)

    print(f"Arquivo de entrada: {RAW_MATCHES_PATH}")
    print(f"Partidas processadas: {len(matches)}")
    print(f"Temporadas: {', '.join(map(str, SEASONS))}")
    print(f"Arquivo de saída: {PROCESSED_MATCHES_PATH}")


if __name__ == "__main__":
    main()
