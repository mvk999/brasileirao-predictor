"""Gera o histórico em Markdown e o relatório visual em PDF.

Uso, na raiz do projeto: python docs/build_report.py
O YAML é a fonte editável; os dois arquivos gerados devem ser versionados juntos.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent
SOURCE = DOCS_DIR / "evolucao.yaml"
MARKDOWN = DOCS_DIR / "EVOLUCAO.md"
PDF = DOCS_DIR / "evolucao-do-projeto.pdf"

NAVY = "#11283E"
INK = "#183047"
MUTED = "#596B7C"
TEAL = "#168C91"
GOLD = "#E6B44E"
PALE = "#F3F7F8"
WHITE = "#FFFFFF"
LINE = "#D9E4E8"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def date_br(value) -> str:
    return value.strftime("%d/%m/%Y")


def percentage(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + "%"


def decimal(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def load_history() -> dict:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    if len(data["dados"]["temporadas"]) != len(data["dados"]["jogos_por_temporada"]):
        raise ValueError("Temporadas e contagens precisam ter o mesmo tamanho.")
    if sum(data["dados"]["jogos_por_temporada"]) != data["dados"]["total_jogos"]:
        raise ValueError("A soma dos jogos por temporada difere do total informado.")
    matrix = np.asarray(data["modelo"]["matriz_confusao_2024"])
    if matrix.shape != (3, 3):
        raise ValueError("A matriz de confusão deve ter três linhas e três colunas.")
    if matrix.sum(axis=1).tolist() != data["modelo"]["reais_2024"]:
        raise ValueError("A matriz de confusão não confere com as classes reais.")
    if matrix.sum(axis=0).tolist() != data["modelo"]["previstas_2024"]:
        raise ValueError("A matriz de confusão não confere com as classes previstas.")
    validation = data["diagnostico_validacao_2023"]
    validation_matrix = np.asarray(validation["matriz_confusao"])
    if validation_matrix.shape != (3, 3):
        raise ValueError("A matriz de validação deve ter três linhas e três colunas.")
    if validation_matrix.sum(axis=1).tolist() != validation["reais"]:
        raise ValueError("A matriz de validação não confere com as classes reais.")
    if validation_matrix.sum(axis=0).tolist() != validation["previstas"]:
        raise ValueError("A matriz de validação não confere com as classes previstas.")
    if sum(band["jogos"] for band in validation["faixas_confianca"]) != 380:
        raise ValueError("As faixas de confiança devem cobrir os 380 jogos de 2023.")
    return data


def verify_local_metrics(data: dict) -> None:
    """Confere o instantâneo se o treino já foi executado nesta máquina."""
    path = ROOT / "artifacts/logistic_baseline_metrics.json"
    if not path.is_file():
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    mapping = {"validacao_2023": "validation", "teste_2024": "test"}
    for section, report_section in mapping.items():
        for label, report_label in (("referencia", "reference"), ("regressao", "logistic_regression")):
            for metric, report_metric in (("acuracia", "accuracy"), ("f1_macro", "f1_macro"), ("log_loss", "log_loss")):
                recorded = data["modelo"]["metricas"][section][label][metric]
                measured = current[report_section][report_label][report_metric]
                if abs(recorded - measured) > 0.000001:
                    raise ValueError(
                        f"Métrica desatualizada em {section}/{label}/{metric}: "
                        f"YAML={recorded}, treino local={measured}"
                    )


def verify_local_diagnostics(data: dict) -> None:
    """Confere a análise documentada com o JSON local, se ele existir."""
    path = ROOT / "artifacts/validation_2023_diagnostics.json"
    if not path.is_file():
        return
    measured = json.loads(path.read_text(encoding="utf-8"))
    recorded = data["diagnostico_validacao_2023"]
    if recorded["matriz_confusao"] != measured["confusion_matrix"]:
        raise ValueError("A matriz de 2023 difere do diagnóstico local.")
    for label, item in recorded["por_classe"].items():
        actual = measured["by_class"][label]
        for key, measured_key in (("reais", "actual"), ("previstas", "predicted"),
                                  ("acertos", "correct")):
            if item[key] != actual[measured_key]:
                raise ValueError(f"Contagem de {label}/{key} difere do diagnóstico local.")
        for key, measured_key in (("precisao", "precision"), ("revocacao", "recall"),
                                  ("f1", "f1")):
            if abs(item[key] - actual[measured_key]) > 0.000001:
                raise ValueError(f"Métrica de {label}/{key} difere do diagnóstico local.")
    if len(recorded["faixas_confianca"]) != len(measured["confidence_bands"]):
        raise ValueError("O número de faixas de confiança difere do diagnóstico local.")
    for item, actual in zip(recorded["faixas_confianca"], measured["confidence_bands"]):
        if item["jogos"] != actual["matches"]:
            raise ValueError("Faixa de confiança difere do diagnóstico local.")
        for key, measured_key in (("inicio", "from"), ("fim", "to"),
                                  ("confianca_media", "mean_confidence")):
            if abs(item[key] - actual[measured_key]) > 0.000001:
                raise ValueError(f"{key} por faixa difere do diagnóstico local.")
        if abs(item["acerto"] - actual["accuracy"]) > 0.000001:
            raise ValueError("Acerto por faixa difere do diagnóstico local.")
    if len(recorded["erros_maior_confianca"]) != len(measured["most_confident_errors"]):
        raise ValueError("O número de exemplos de erro difere do diagnóstico local.")
    for item, actual in zip(recorded["erros_maior_confianca"], measured["most_confident_errors"]):
        for key, measured_key in (("id", "id"), ("mandante", "home"),
                                  ("visitante", "away"), ("real", "actual"),
                                  ("previsto", "predicted")):
            if item[key] != actual[measured_key]:
                raise ValueError(f"Exemplo de erro {item['id']} difere em {key}.")
        if str(item["data"]) != actual["date"]:
            raise ValueError(f"Data do erro {item['id']} difere do diagnóstico local.")
        if abs(item["confianca"] - actual["confidence"]) > 0.000001:
            raise ValueError(f"Confiança do erro {item['id']} difere do diagnóstico local.")


def verify_local_draw_experiment(data: dict) -> None:
    """Evita publicar números do experimento diferentes do relatório gerado."""
    path = ROOT / "artifacts/draw_experiment.json"
    if not path.is_file():
        return
    measured = json.loads(path.read_text(encoding="utf-8"))["feature_sets"]
    recorded = data["experimento_empates"]
    mapping = (
        ("referencia_padrao", measured["baseline"]["validation_2023_default"]),
        ("referencia_ajustada", measured["baseline"]["validation_2023_adjusted"]),
        ("equilibrio_ajustado", measured["balance_features"]["validation_2023_adjusted"]),
    )
    for name, actual in mapping:
        row = recorded[name]
        for documented, generated in (("acuracia", "accuracy"), ("f1_macro", "f1_macro"),
                                      ("log_loss", "log_loss"), ("precisao_d", "draw_precision"),
                                      ("revocacao_d", "draw_recall")):
            if abs(row[documented] - actual[generated]) > 0.000001:
                raise ValueError(f"Experimento desatualizado em {name}/{documented}.")
        for documented, generated in (("previstos_d", "draw_predicted"), ("corretos_d", "draw_correct")):
            if row[documented] != actual[generated]:
                raise ValueError(f"Experimento desatualizado em {name}/{documented}.")
    for name, key in (("referencia_ajustada", "baseline"), ("equilibrio_ajustado", "balance_features")):
        if recorded[name]["limiar"] != measured[key]["threshold"]:
            raise ValueError(f"Limiar desatualizado em {name}.")
    for row in recorded["troca_limiares"]:
        actual = measured["baseline"]["validation_2023_illustrative"][f"{row['limiar']:.2f}"]
        if (row["previstos_d"] != actual["draw_predicted"]
                or row["corretos_d"] != actual["draw_correct"]
                or row["falsos_d"] != actual["draw_predicted"] - actual["draw_correct"]
                or abs(row["acuracia"] - actual["accuracy"]) > 0.000001):
            raise ValueError(f"Troca de erros desatualizada no limiar {row['limiar']}.")


def verify_local_poisson_experiment(data: dict) -> None:
    path = ROOT / "artifacts/poisson_experiment.json"
    if not path.is_file():
        return
    measured = json.loads(path.read_text(encoding="utf-8"))
    recorded = data["experimento_poisson"]
    if recorded["prior_escolhido"] != measured["chosen_prior_games"]:
        raise ValueError("Prior do Poisson difere do relatório local.")
    for name, key in (("referencia", "logistic_baseline"), ("poisson", "poisson")):
        actual = measured["validation_2023"][key]
        row = recorded[name]
        for documented, generated in (("acuracia", "accuracy"), ("f1_macro", "f1_macro"),
                                      ("log_loss", "log_loss")):
            if abs(row[documented] - actual[generated]) > 0.000001:
                raise ValueError(f"Experimento Poisson desatualizado em {name}/{documented}.")
        for label, prefix in (("D", "d"), ("H", "h"), ("A", "a")):
            if row[f"{prefix}_corretos"] != actual["by_class"][label]["correct"]:
                raise ValueError(f"Acertos de {label} diferem no experimento Poisson.")
        if row["d_previstos"] != actual["by_class"]["D"]["predicted"]:
            raise ValueError("Empates previstos diferem no experimento Poisson.")


def verify_local_round_experiment(data: dict) -> None:
    path = ROOT / "artifacts/round_draw_experiment.json"
    if not path.is_file():
        return
    measured = json.loads(path.read_text(encoding="utf-8"))
    recorded = data["experimento_rodadas"]
    if (recorded["limite_gols"] != measured["chosen_gap_limit"]
            or recorded["teto_empates"] != measured["max_draws_per_round"]
            or recorded["empates_exatamente_iguais_2023"] != measured["equal_estimated_goals_2023"]
            or recorded["max_empates_reais_rodada_2023"] != measured["actual_draws_per_round_2023"]["maximum"]
            or recorded["rodadas_acima_teto_2023"] != measured["actual_draws_per_round_2023"]["rounds_above_cap"]
            or recorded["rodadas_acima_teto_numeros_2023"] != measured["actual_draws_per_round_2023"]["round_numbers_above_cap"]):
        raise ValueError("Configuração da regra por rodada difere do relatório local.")
    for name, key in (("referencia", "poisson_argmax"), ("regra", "round_rule")):
        row = recorded[name]
        actual = measured["validation_2023"][key]
        for documented, generated in (("acuracia", "accuracy"), ("f1_macro", "f1_macro"),
                                      ("log_loss", "log_loss")):
            if abs(row[documented] - actual[generated]) > 0.000001:
                raise ValueError(f"Regra por rodada desatualizada em {name}/{documented}.")
        for label, prefix in (("D", "d"), ("H", "h"), ("A", "a")):
            if row[f"{prefix}_corretos"] != actual["by_class"][label]["correct"]:
                raise ValueError(f"Acertos de {label} diferem na regra por rodada.")
        if row["d_previstos"] != actual["by_class"]["D"]["predicted"]:
            raise ValueError("Empates previstos diferem na regra por rodada.")


def verify_local_long_history_experiment(data: dict) -> None:
    path = ROOT / "artifacts/long_history_experiment.json"
    if not path.is_file():
        return
    measured = json.loads(path.read_text(encoding="utf-8"))
    recorded = data["experimento_historico_longo"]
    if (recorded["temporadas_completas"] != len(measured["source_years"])
            or recorded["partidas"] != measured["source_games"]):
        raise ValueError("Cobertura do histórico longo difere do relatório local.")
    for section, actual_section in (("metricas_medias_2018_2022", "backtest_average"),
                                    ("validacao_2023", "retrospective"),
                                    ("avaliacao_2024", "retrospective")):
        for strategy, row in recorded[section].items():
            actual = measured[actual_section][strategy] if actual_section == "backtest_average" else measured[actual_section][section[-4:]][strategy]
            for documented, generated in (("acuracia", "accuracy"), ("f1_macro", "f1_macro"),
                                          ("log_loss", "log_loss")):
                if abs(row[documented] - actual[generated]) > 0.000001:
                    raise ValueError(f"Histórico longo desatualizado em {section}/{strategy}/{documented}.")
            if "empates_corretos" in row:
                if (row["empates_corretos"] != actual["by_class"]["D"]["correct"]
                        or row["empates_previstos"] != actual["by_class"]["D"]["predicted"]):
                    raise ValueError(f"Empates diferem em {section}/{strategy}.")


def make_markdown(data: dict) -> str:
    model = data["modelo"]
    metrics = model["metricas"]
    validation = data["diagnostico_validacao_2023"]
    total_games = f"{data['dados']['total_jogos']:,}".replace(",", ".")
    long_history_games = f"{data['experimento_historico_longo']['partidas']:,}".replace(",", ".")
    lines = [
        f"# Evolução do {data['titulo']}",
        "",
        f"**Edição {data['edicao']} · Atualizado em {date_br(data['atualizado_em'])}**",
        "",
        "Este é o registro cronológico do projeto. `evolucao.yaml` contém os fatos",
        "editáveis; `EVOLUCAO.md` e `evolucao-do-projeto.pdf` são gerados a partir dele.",
        "",
        "## Como começou",
        "",
        data["origem"]["descricao"],
        "",
        f"Evidência inicial: commit `{data['origem']['commit']}` de {date_br(data['origem']['data'])}.",
        "",
        "## Linha do tempo",
        "",
        "| Data | Marco | O que passou a existir | Evidência |",
        "| --- | --- | --- | --- |",
    ]
    for item in data["marcos"]:
        lines.append(
            f"| {date_br(item['data'])} | {item['titulo']} | {item['detalhe']} | "
            f"`{item['evidencia']}` |"
        )
    lines += [
        "",
        "## Estado atual dos dados",
        "",
        f"O recorte contém **{total_games} jogos** de "
        f"{data['dados']['temporadas'][0]} a {data['dados']['temporadas'][-1]}, "
        f"com **{data['dados']['features_por_partida']} features por partida** "
        f"em uma tabela de {data['dados']['colunas_base_features']} colunas.",
        "",
        "| Temporada | Partidas |",
        "| --- | ---: |",
    ]
    for year, count in zip(data["dados"]["temporadas"], data["dados"]["jogos_por_temporada"]):
        lines.append(f"| {year} | {count} |")
    lines += [
        "",
        f"O alvo é `{data['dados']['alvo']}`. {data['dados']['nota']}",
        "",
        "```text",
        "CSV histórico → preparação → exploração → features por partida",
        "             → treino temporal → avaliação retrospectiva de 2024",
        "```",
        "",
        "## Primeiro modelo e avaliação",
        "",
        f"**Modelo:** {model['tipo']}. **Entrada:** {model['entrada']}. "
        f"**Preparo:** {model['preparo']}.",
        "",
        f"**Separação:** treino {model['treino']}; validação {model['validacao']}; "
        f"teste {model['teste']}. {model['retreino']}",
        "",
        f"**Referência de comparação:** {model['referencia']}.",
        "",
        "| Período | Método | Acurácia | F1 macro | Log loss |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, period in (("validacao_2023", "Validação 2023"), ("teste_2024", "Teste 2024")):
        for label, name in (("referencia", "Referência"), ("regressao", "Regressão logística")):
            row = metrics[key][label]
            lines.append(
                f"| {period} | {name} | {percentage(row['acuracia'])} | "
                f"{decimal(row['f1_macro'])} | {decimal(row['log_loss'])} |"
            )
    lines += [
        "",
        "O teste de 2024 é retrospectivo. As features de cada partida podem usar",
        "resultados **anteriores** da mesma temporada; não usam o resultado da",
        "partida avaliada. O desempenho ainda é limitado: a regressão previu",
        f"**{model['previstas_2024'][2]} vitórias do mandante** em 380 jogos,",
        f"mas só **{model['previstas_2024'][1]} empates**. A matriz abaixo usa",
        "linhas para o resultado real e colunas para a previsão.",
        "",
        "| Real / Previsto | A | D | H |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, row in zip(model["classes"], model["matriz_confusao_2024"]):
        lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")
    lines += [
        "",
        "## Diagnóstico dos erros na validação de 2023",
        "",
        validation["treino"],
        "",
        "A matriz usa **linhas para o resultado real** e **colunas para a previsão**:",
        "",
        "| Real / Previsto | A | D | H |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, row in zip(validation["classes"], validation["matriz_confusao"]):
        lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")
    lines += [
        "",
        "| Classe | Reais | Previstos | Acertos | Precisão | Revocação | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in validation["classes"]:
        item = validation["por_classe"][label]
        lines.append(
            f"| {label} | {item['reais']} | {item['previstas']} | {item['acertos']} | "
            f"{percentage(item['precisao'])} | {percentage(item['revocacao'])} | "
            f"{decimal(item['f1'])} |"
        )
    draw = validation["por_classe"]["D"]
    home = validation["por_classe"]["H"]
    lines += [
        "",
        f"O modelo reconheceu **{draw['acertos']} dos {draw['reais']} empates** "
        f"({percentage(draw['revocacao'])} de revocação), pois só escolheu D "
        f"em {draw['previstas']} jogos. Em contraste, acertou "
        f"**{home['acertos']} das {home['reais']} vitórias do mandante** "
        f"({percentage(home['revocacao'])} de revocação), mas escolheu H "
        f"em {home['previstas']} jogos. Isso mostra concentração das previsões "
        "em H; não demonstra, por si só, a causa do comportamento.",
        "",
        "### Confiança e acerto observado",
        "",
        "A confiança é a maior das três probabilidades previstas. Cada faixa",
        "compara a confiança média com a proporção de acertos nas partidas nela",
        "incluídas. Faixas pequenas exigem cautela na interpretação.",
        "",
        "| Confiança prevista | Jogos | Confiança média | Acerto observado |",
        "| --- | ---: | ---: | ---: |",
    ]
    for band in validation["faixas_confianca"]:
        lines.append(
            f"| {percentage(band['inicio'])}–{percentage(band['fim'])} | "
            f"{band['jogos']} | {percentage(band['confianca_media'])} | "
            f"{percentage(band['acerto'])} |"
        )
    lines += [
        "",
        "### Cinco erros com maior confiança na classe prevista",
        "",
        "Esses exemplos ajudam a inspecionar o comportamento do modelo; não",
        "explicam isoladamente por que ele errou.",
        "",
        "| Data | Partida | Real | Previsto | Confiança |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for item in validation["erros_maior_confianca"]:
        lines.append(
            f"| {date_br(item['data'])} | {item['mandante']} × {item['visitante']} "
            f"(ID {item['id']}) | {item['real']} | {item['previsto']} | "
            f"{percentage(item['confianca'])} |"
        )
    experiment = data["experimento_empates"]
    lines += [
        "",
        "## Experimento com empates",
        "",
        experiment["protocolo"],
        "",
        "A regra ajustada escolhe empate quando sua probabilidade atinge o",
        "limite; caso contrário, escolhe entre mandante e visitante. A tabela",
        "mostra o desempenho em 2023. A mudança de limite não altera as",
        "probabilidades, então o log loss da mesma linha de modelo não muda.",
        "",
        "| Método | Limite D | Acurácia | F1 macro | Log loss | D previstos | D corretos | Precisão D | Revocação D |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, title in (("referencia_padrao", "Referência original"),
                       ("referencia_ajustada", "Referência com limite"),
                       ("equilibrio_ajustado", "Equilíbrio com limite")):
        row = experiment[key]
        lines.append(
            f"| {title} | {row['limiar']} | {percentage(row['acuracia'])} | "
            f"{decimal(row['f1_macro'])} | {decimal(row['log_loss'])} | "
            f"{row['previstos_d']} | {row['corretos_d']} | "
            f"{percentage(row['precisao_d'])} | {percentage(row['revocacao_d'])} |"
        )
    lines += [
        "",
        "**Novos sinais testados:** " + experiment["features"],
        "",
        "### Troca entre empates encontrados e falsos empates",
        "",
        "Os limites abaixo são apenas ilustrativos em 2023, com o modelo de 12 features.",
        "",
        "| Limite | Empates previstos | Empates corretos | Falsos empates | Acurácia |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in experiment["troca_limiares"]:
        lines.append(
            f"| {row['limiar']:.2f} | {row['previstos_d']} | {row['corretos_d']} | "
            f"{row['falsos_d']} | {percentage(row['acuracia'])} |"
        )
    poisson = data["experimento_poisson"]
    lines += [
        "", "**Conclusão:** " + experiment["interpretacao"],
        "", "## Experimento com gols por Poisson", "", poisson["protocolo"],
        "", poisson["metodo"], "",
        "| Método em 2023 | Acurácia | F1 macro | Log loss | Empates corretos / previstos | H corretos / reais | A corretos / reais |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, name in (("referencia", "Regressão logística"), ("poisson", "Poisson independente")):
        row = poisson[key]
        lines.append(
            f"| {name} | {percentage(row['acuracia'])} | {decimal(row['f1_macro'])} | "
            f"{decimal(row['log_loss'])} | {row['d_corretos']}/{row['d_previstos']} | "
            f"{row['h_corretos']}/178 | {row['a_corretos']}/104 |"
        )
    round_experiment = data["experimento_rodadas"]
    lines += [
        "", "**Conclusão:** " + poisson["conclusao"],
        "", "## Experimento: até cinco empates por rodada", "",
        round_experiment["protocolo"], "",
        "As taxas de gols vêm de placares históricos; elas não são xG de finalizações.",
        f"A diferença escolhida em 2022 foi de {round_experiment['limite_gols']:.2f} gol.",
        "", "| Método em 2023 | Acurácia | F1 macro | Empates corretos / previstos | H corretos / reais | A corretos / reais |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, title in (("referencia", "Poisson por rodada"), ("regra", "Proximidade + teto")):
        row = round_experiment[key]
        lines.append(
            f"| {title} | {percentage(row['acuracia'])} | {decimal(row['f1_macro'])} | "
            f"{row['d_corretos']}/{row['d_previstos']} | "
            f"{row['h_corretos']}/178 | {row['a_corretos']}/104 |"
        )
    long_history = data["experimento_historico_longo"]
    lines += [
        "", "**Conclusão:** " + round_experiment["conclusao"],
        "", "## Experimento: aprender com mais temporadas", "",
        long_history["protocolo"], "", long_history["ausencias"], "",
        f"A base experimental contém **{long_history_games} jogos** em "
        f"**{long_history['temporadas_completas']} temporadas completas**, com os mesmos 12 atributos.",
        "", "| Treino anterior ao ano avaliado | Acurácia média 2018–2022 | F1 macro médio | Log loss médio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, name in (("recent_3", "3 temporadas"), ("recent_5", "5 temporadas"),
                      ("recent_10", "10 temporadas"), ("all", "Todo o histórico")):
        row = long_history["metricas_medias_2018_2022"][key]
        lines.append(
            f"| {name} | {percentage(row['acuracia'])} | "
            f"{decimal(row['f1_macro'])} | {decimal(row['log_loss'])} |"
        )
    lines += [
        "", "| Ano | Treino | Acurácia | F1 macro | Log loss | Empates corretos / previstos |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for year, section in ((2023, "validacao_2023"), (2024, "avaliacao_2024")):
        for key, name in (("recent_3", "3 temporadas"), ("all", "Todo o histórico")):
            row = long_history[section][key]
            lines.append(
                f"| {year} | {name} | {percentage(row['acuracia'])} | "
                f"{decimal(row['f1_macro'])} | {decimal(row['log_loss'])} | "
                f"{row['empates_corretos']}/{row['empates_previstos']} |"
            )
    lines += ["", "**Conclusão:** " + long_history["conclusao"], "", "## O que existe hoje", ""]
    lines += [f"- {item}" for item in data["estado_atual"]["entregue"]]
    lines += ["", "## Limites conhecidos", ""]
    lines += [f"- {item}" for item in data["estado_atual"]["limites"]]
    lines += ["", "## Próximas etapas", ""]
    lines += [f"{index}. {item}" for index, item in enumerate(data["estado_atual"]["proximos_passos"], 1)]
    lines += [
        "",
        "## Como manter este histórico",
        "",
        "1. Ao concluir uma mudança, adicione um marco datado em `docs/evolucao.yaml`",
        "   com o arquivo ou commit que comprova o que foi feito.",
        "2. Atualize números e conclusões apenas depois de gerar e conferir as",
        "   saídas do projeto. Registre limites e trabalho pendente.",
        "3. Na raiz do repositório, execute `python docs/build_report.py` e revise",
        "   o diff de `docs/EVOLUCAO.md` e o PDF antes de publicar.",
        "4. Versione juntos o YAML, o Markdown e o PDF.",
        "",
        "## Fontes internas desta edição",
        "",
    ]
    lines += [f"- {item}" for item in data["fontes"]]
    return "\n".join(lines) + "\n"


def box(fig, x, y, width, height, *, face=WHITE, edge=LINE, radius=0.02):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure,
        linewidth=0.8,
        facecolor=face,
        edgecolor=edge,
        zorder=1,
    )
    fig.patches.append(patch)


def label(fig, x, y, text, *, size=10, color=INK, weight="normal", ha="left", va="top"):
    fig.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5)


def wrapped(fig, x, y, text, width, *, size=10, color=MUTED, line_spacing=1.35):
    fig.text(
        x, y, fill(text, width=width), fontsize=size, color=color,
        va="top", linespacing=line_spacing, zorder=5,
    )


def page(title: str, kicker: str, number: int, total: int, *, dark=False):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=NAVY if dark else PALE)
    fig.patches.append(
        Rectangle((0, 0.985), 1, 0.015, transform=fig.transFigure, facecolor=GOLD, edgecolor="none")
    )
    if not dark:
        label(fig, 0.06, 0.935, kicker.upper(), size=9, color=TEAL, weight="bold")
        label(fig, 0.06, 0.895, title, size=21, color=NAVY, weight="bold")
        fig.lines.append(
            mpl.lines.Line2D([0.06, 0.94], [0.855, 0.855], transform=fig.transFigure, color=LINE, lw=1)
        )
    label(fig, 0.06, 0.045, "BRASILEIRÃO PREDICTOR  •  EVOLUÇÃO DO PROJETO", size=8,
          color="#B4C4D0" if dark else MUTED, weight="bold")
    label(fig, 0.94, 0.045, f"{number:02d} / {total:02d}", size=8,
          color="#B4C4D0" if dark else MUTED, weight="bold", ha="right")
    return fig


def cover(data: dict, total: int):
    fig = page("", "", 1, total, dark=True)
    # Campo de futebol discreto ao fundo: metáfora visual, sem dado inventado.
    fig.patches.append(Rectangle((0.72, 0.18), 0.23, 0.56, transform=fig.transFigure,
                                 fill=False, lw=1.5, edgecolor="#35536A"))
    fig.patches.append(Circle((0.835, 0.46), 0.060, transform=fig.transFigure,
                              fill=False, lw=1.5, edgecolor="#35536A"))
    fig.lines.append(mpl.lines.Line2D([0.72, 0.95], [0.46, 0.46], transform=fig.transFigure,
                                       color="#35536A", lw=1.5))
    label(fig, 0.07, 0.86, f"RELATÓRIO DE EVOLUÇÃO  ·  EDIÇÃO {data['edicao']:02d}",
          size=11, color=GOLD, weight="bold")
    label(fig, 0.07, 0.76, "Brasileirão", size=32, color=WHITE, weight="bold")
    label(fig, 0.07, 0.675, "Predictor", size=32, color=WHITE, weight="bold")
    wrapped(fig, 0.07, 0.57, data["subtitulo"], 45, size=13, color="#C9D8DF")
    label(fig, 0.07, 0.48, f"Atualizado em {date_br(data['atualizado_em'])}", size=10, color="#B5C8D1")

    values = ["5", "1.900", "12", percentage(data["modelo"]["metricas"]["teste_2024"]["regressao"]["acuracia"])]
    captions = ["TEMPORADAS", "JOGOS VALIDADOS", "FEATURES / JOGO", "ACURÁCIA 2024"]
    for i, (value, caption) in enumerate(zip(values, captions)):
        x = 0.07 + i * 0.22
        box(fig, x, 0.21, 0.195, 0.16, face="#1D3A50", edge="#31566A")
        label(fig, x + 0.02, 0.325, value, size=24, color=WHITE, weight="bold")
        label(fig, x + 0.02, 0.245, caption, size=8, color="#AFCDD4", weight="bold")
    label(fig, 0.07, 0.12, "DA IDEIA → À BASE DE DADOS → À PRIMEIRA AVALIAÇÃO", size=10,
          color=GOLD, weight="bold")
    return fig


def history_page(data: dict, items: list[dict], number: int, total: int):
    fig = page("Como o projeto chegou até aqui", "Trajetória documentada", number, total)
    fig.lines.append(mpl.lines.Line2D([0.105, 0.105], [0.17, 0.79], transform=fig.transFigure,
                                       color=TEAL, lw=2.2))
    for index, item in enumerate(items):
        y = 0.79 - index * (0.095 if len(items) > 6 else 0.11)
        fig.patches.append(Circle((0.105, y - 0.005), 0.010, transform=fig.transFigure,
                                  color=TEAL if index == len(items) - 1 else WHITE, ec=TEAL, lw=2))
        label(fig, 0.065, y + 0.016, date_br(item["data"]), size=8, color=MUTED, ha="center")
        label(fig, 0.14, y + 0.012, item["titulo"], size=11, color=INK, weight="bold")
        wrapped(fig, 0.14, y - 0.019, item["detalhe"], 68, size=9, color=MUTED)
        label(fig, 0.58, y + 0.013, item["evidencia"], size=8, color=TEAL, ha="right")

    box(fig, 0.63, 0.51, 0.31, 0.29)
    label(fig, 0.66, 0.755, "O PONTO DE PARTIDA", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.66, 0.70, data["origem"]["descricao"], 39, size=10, color=INK)
    box(fig, 0.63, 0.18, 0.31, 0.28, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.66, 0.42, "O QUE JÁ É CONCRETO", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.66, 0.36,
            "CSV preparado; 12 métricas por partida; modelo avaliado em 2024; erros de 2023 analisados.",
            39, size=11, color=INK)
    label(fig, 0.66, 0.23, "Objetivos iniciais e entregas têm estados distintos.", size=8, color=MUTED)
    return fig


def data_page(data: dict, number: int, total: int):
    fig = page("Da fonte bruta à base de features", "Dados e engenharia", number, total)
    d = data["dados"]
    box(fig, 0.06, 0.46, 0.42, 0.34)
    label(fig, 0.085, 0.765, "COBERTURA POR TEMPORADA", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.10, 0.525, 0.34, 0.195], facecolor=WHITE)
    ax.set_zorder(3)
    colors = ["#BBDAD8", "#A5D0CE", "#8AC3C0", "#63ADA9", TEAL]
    ax.bar([str(year) for year in d["temporadas"]], d["jogos_por_temporada"], color=colors, width=0.62)
    ax.set_ylim(0, 455)
    ax.set_yticks([0, 190, 380])
    ax.tick_params(axis="both", labelsize=9, length=0)
    ax.grid(axis="y", color=LINE, alpha=0.7)
    ax.set_axisbelow(True)
    for i, count in enumerate(d["jogos_por_temporada"]):
        ax.text(i, count + 11, str(count), ha="center", va="bottom", color=INK, fontsize=8, fontweight="bold")

    box(fig, 0.52, 0.46, 0.42, 0.34)
    label(fig, 0.55, 0.765, "TRANSFORMAÇÃO", size=10, color=TEAL, weight="bold")
    stages = [("CSV bruto", "fonte pública"), ("Preparação", "2020–2024"),
              ("Features", "12 por partida"), ("Avaliação", "2024")]
    for i, (title, subtitle) in enumerate(stages):
        y = 0.70 - i * 0.063
        box(fig, 0.55, y - 0.042, 0.34, 0.054, face=PALE, edge=LINE, radius=0.01)
        label(fig, 0.57, y - 0.001, title, size=10, color=INK, weight="bold")
        label(fig, 0.865, y - 0.001, subtitle, size=8, color=MUTED, ha="right")
        if i < 3:
            label(fig, 0.72, y - 0.044, "↓", size=11, color=TEAL, ha="center")

    box(fig, 0.06, 0.17, 0.88, 0.23)
    label(fig, 0.085, 0.355, "REGRAS QUE IMPORTAM", size=10, color=TEAL, weight="bold")
    columns = [
        ("1.900", "jogos validados", "380 por temporada, 20 clubes e 38 rodadas"),
        ("12", "features históricas", "6 do mandante + 6 do visitante"),
        ("H / D / A", "alvo", "mandante / empate / visitante"),
    ]
    for i, (value, title, detail) in enumerate(columns):
        x = 0.085 + i * 0.285
        label(fig, x, 0.305, value, size=18, color=NAVY, weight="bold")
        label(fig, x, 0.255, title.upper(), size=8, color=TEAL, weight="bold")
        wrapped(fig, x, 0.225, detail, 34, size=8, color=MUTED)
    return fig


def model_page(data: dict, number: int, total: int):
    fig = page("Primeira avaliação: melhora pequena, limite claro", "Modelo e evidência", number, total)
    m = data["modelo"]
    rows = m["metricas"]["teste_2024"]

    box(fig, 0.06, 0.42, 0.42, 0.39)
    label(fig, 0.085, 0.77, "TESTE FINAL · 2024", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.12, 0.51, 0.31, 0.20], facecolor=WHITE)
    ax.set_zorder(3)
    labels = ["Acurácia", "F1 macro"]
    reference = [rows["referencia"]["acuracia"], rows["referencia"]["f1_macro"]]
    logistic = [rows["regressao"]["acuracia"], rows["regressao"]["f1_macro"]]
    x = np.arange(2)
    ax.bar(x - 0.17, reference, 0.30, color="#B5C7CF", label="Referência")
    ax.bar(x + 0.17, logistic, 0.30, color=TEAL, label="Regressão")
    ax.set_ylim(0, 0.62)
    ax.set_xticks(x, labels)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    for i, value in enumerate(reference):
        ax.text(i - 0.17, value + 0.015, f"{value:.3f}", ha="center", size=7, color=MUTED)
    for i, value in enumerate(logistic):
        ax.text(i + 0.17, value + 0.015, f"{value:.3f}", ha="center", size=7, color=TEAL, fontweight="bold")

    box(fig, 0.52, 0.42, 0.42, 0.39)
    label(fig, 0.55, 0.77, "ONDE O MODELO ACERTA E ERRA", size=10, color=TEAL, weight="bold")
    ax2 = fig.add_axes([0.61, 0.53, 0.25, 0.22], facecolor=WHITE)
    ax2.set_zorder(3)
    matrix = np.asarray(m["matriz_confusao_2024"])
    ax2.imshow(matrix, cmap=mpl.colors.LinearSegmentedColormap.from_list("teal", [WHITE, TEAL]), vmin=0, vmax=165)
    ax2.set_xticks(range(3), m["classes"])
    ax2.set_yticks(range(3), m["classes"])
    ax2.set_xlabel("Previsto", fontsize=8)
    ax2.set_ylabel("Real", fontsize=8)
    ax2.tick_params(length=0, labelsize=9)
    for (row, col), value in np.ndenumerate(matrix):
        ax2.text(col, row, str(value), ha="center", va="center", fontsize=10,
                 color=WHITE if value > 80 else INK, fontweight="bold")
    label(fig, 0.55, 0.46, "A: visitante   D: empate   H: mandante", size=8, color=MUTED)

    box(fig, 0.06, 0.14, 0.88, 0.24, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.345, "MÉTRICAS 2024", size=10, color=TEAL, weight="bold")
    label(fig, 0.265, 0.305, "Referência", size=8, color=MUTED, ha="center")
    label(fig, 0.405, 0.305, "Regressão", size=8, color=MUTED, ha="center")
    for y, title, metric in ((0.27, "Acurácia", "acuracia"),
                              (0.235, "F1 macro", "f1_macro"),
                              (0.20, "Log loss", "log_loss")):
        label(fig, 0.085, y, title, size=9, color=INK)
        left = percentage(rows["referencia"][metric]) if metric == "acuracia" else decimal(rows["referencia"][metric])
        right = percentage(rows["regressao"][metric]) if metric == "acuracia" else decimal(rows["regressao"][metric])
        label(fig, 0.265, y, left, size=9, color=MUTED, ha="center")
        label(fig, 0.405, y, right, size=9, color=TEAL, weight="bold", ha="center")
    fig.lines.append(mpl.lines.Line2D([0.49, 0.49], [0.17, 0.35], transform=fig.transFigure,
                                       color="#C3E3E0", lw=1))
    label(fig, 0.52, 0.345, "LEITURA HONESTA", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.52, 0.30,
            "A regressão acertou 183 de 380 jogos, três a mais que a referência. "
            "Ela previu vitória do mandante em 323 jogos e empate em apenas 17. "
            "É um teste retrospectivo; ainda não há previsão de jogos futuros.",
            60, size=9, color=INK)
    return fig


def validation_page(data: dict, number: int, total: int):
    fig = page("Validação de 2023: o empate quase não aparece", "Erros por resultado", number, total)
    validation = data["diagnostico_validacao_2023"]
    classes = validation["classes"]

    box(fig, 0.06, 0.43, 0.42, 0.38)
    label(fig, 0.085, 0.77, "DISTRIBUIÇÃO REAL × PREVISTA", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.12, 0.51, 0.31, 0.22], facecolor=WHITE)
    ax.set_zorder(3)
    x = np.arange(3)
    ax.bar(x - 0.17, validation["reais"], 0.30, label="Real", color="#B5C7CF")
    ax.bar(x + 0.17, validation["previstas"], 0.30, label="Previsto", color=TEAL)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 100, 200, 300])
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for i, (actual, predicted) in enumerate(zip(validation["reais"], validation["previstas"])):
        ax.text(i - 0.17, actual + 6, str(actual), ha="center", fontsize=8, color=MUTED)
        ax.text(i + 0.17, predicted + 6, str(predicted), ha="center", fontsize=8,
                color=TEAL, fontweight="bold")

    box(fig, 0.52, 0.43, 0.42, 0.38)
    label(fig, 0.55, 0.77, "MATRIZ DE CONFUSÃO", size=10, color=TEAL, weight="bold")
    ax2 = fig.add_axes([0.61, 0.53, 0.25, 0.22], facecolor=WHITE)
    ax2.set_zorder(3)
    matrix = np.asarray(validation["matriz_confusao"])
    ax2.imshow(matrix, cmap=mpl.colors.LinearSegmentedColormap.from_list("teal23", [WHITE, TEAL]),
               vmin=0, vmax=170)
    ax2.set_xticks(range(3), classes)
    ax2.set_yticks(range(3), classes)
    ax2.set_xlabel("Previsto", fontsize=8)
    ax2.set_ylabel("Real", fontsize=8)
    ax2.tick_params(length=0, labelsize=9)
    for (row, col), value in np.ndenumerate(matrix):
        ax2.text(col, row, str(value), ha="center", va="center", fontsize=10,
                 color=WHITE if value > 80 else INK, fontweight="bold")
    label(fig, 0.55, 0.46, "A: visitante   D: empate   H: mandante", size=8, color=MUTED)

    box(fig, 0.06, 0.14, 0.88, 0.24, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.345, "RESULTADO POR CLASSE", size=10, color=TEAL, weight="bold")
    headings = [(0.09, "Classe"), (0.22, "Reais"), (0.34, "Previstos"),
                (0.47, "Acertos"), (0.60, "Precisão"), (0.73, "Revocação"), (0.86, "F1")]
    for x, heading in headings:
        label(fig, x, 0.305, heading, size=8, color=MUTED, ha="center")
    for index, class_name in enumerate(classes):
        row = validation["por_classe"][class_name]
        values = [class_name, str(row["reais"]), str(row["previstas"]), str(row["acertos"]),
                  percentage(row["precisao"]), percentage(row["revocacao"]), decimal(row["f1"])]
        for (x, _), value in zip(headings, values):
            label(fig, x, 0.267 - index * 0.039, value, size=9,
                  color=TEAL if class_name == "D" else INK,
                  weight="bold" if class_name == "D" else "normal", ha="center")
    return fig


def confidence_page(data: dict, number: int, total: int):
    fig = page("Confiança prevista e erros concretos", "Diagnóstico de 2023", number, total)
    validation = data["diagnostico_validacao_2023"]
    bands = validation["faixas_confianca"]

    box(fig, 0.06, 0.40, 0.42, 0.41)
    label(fig, 0.085, 0.77, "CONFIANÇA × ACERTO OBSERVADO", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.10, 0.49, 0.34, 0.22], facecolor=WHITE)
    ax.set_zorder(3)
    x = np.arange(len(bands))
    ax.plot(x, [item["confianca_media"] for item in bands], color="#93ACB6",
            marker="o", lw=2, label="Confiança média")
    ax.plot(x, [item["acerto"] for item in bands], color=TEAL,
            marker="o", lw=2, label="Acerto observado")
    ax.set_xticks(x, [f"{int(item['inicio'] * 100)}–{int(item['fim'] * 100)}%\nn={item['jogos']}"
                      for item in bands])
    ax.set_ylim(0.25, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="both", labelsize=7, length=0)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    box(fig, 0.52, 0.40, 0.42, 0.41)
    label(fig, 0.55, 0.77, "ERROS COM MAIOR CONFIANÇA", size=10, color=TEAL, weight="bold")
    for index, item in enumerate(validation["erros_maior_confianca"]):
        y = 0.71 - index * 0.061
        label(fig, 0.55, y, f"{index + 1:02d}", size=8, color=TEAL, weight="bold")
        label(fig, 0.59, y, f"{item['mandante']} × {item['visitante']}", size=8, color=INK, weight="bold")
        label(fig, 0.59, y - 0.025,
              f"{date_br(item['data'])}  ·  {item['real']} → {item['previsto']}  ·  {percentage(item['confianca'])}",
              size=7, color=MUTED)

    box(fig, 0.06, 0.15, 0.88, 0.20, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.315, "COMO INTERPRETAR", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.085, 0.27,
            "Entre 50% e 70% de confiança, o acerto observado ficou abaixo da confiança média. "
            "Acima de 70% há apenas dois jogos, então a faixa é pequena demais para uma conclusão geral. "
            "Os cinco erros destacados mostram casos para investigar; não identificam sua causa. "
            "O teste de 2024 continua reservado como registro final desta versão.",
            121, size=9, color=INK)
    return fig


def draw_experiment_page(data: dict, number: int, total: int):
    fig = page("Empates: medir o ganho e o custo", "Experimento temporal", number, total)
    experiment = data["experimento_empates"]
    rows = experiment["troca_limiares"]

    box(fig, 0.06, 0.41, 0.42, 0.40)
    label(fig, 0.085, 0.77, "LIMITE × ACERTOS E FALSOS EMPATES", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.11, 0.50, 0.32, 0.22], facecolor=WHITE)
    ax.set_zorder(3)
    x = np.arange(len(rows))
    ax.bar(x, [item["corretos_d"] for item in rows], color=TEAL, label="Corretos")
    ax.bar(x, [item["falsos_d"] for item in rows], bottom=[item["corretos_d"] for item in rows],
           color="#B5C7CF", label="Falsos")
    ax.set_xticks(x, [f"{item['limiar']:.2f}" for item in rows])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylim(0, 335)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    for i, item in enumerate(rows):
        ax.text(i, item["previstos_d"] + 7, str(item["previstos_d"]), ha="center", size=8, color=INK)
    label(fig, 0.12, 0.465, "Limiar de probabilidade de empate", size=8, color=MUTED)

    box(fig, 0.52, 0.41, 0.42, 0.40)
    label(fig, 0.55, 0.77, "COMPARAÇÃO EM 2023", size=10, color=TEAL, weight="bold")
    names = (("referencia_padrao", "Modelo original"),
             ("referencia_ajustada", "Limite ajustado"),
             ("equilibrio_ajustado", "Equilíbrio + limite"))
    label(fig, 0.55, 0.69, "MÉTODO", size=8, color=MUTED)
    label(fig, 0.76, 0.69, "F1 MACRO", size=8, color=MUTED, ha="center")
    label(fig, 0.88, 0.69, "EMPATES", size=8, color=MUTED, ha="center")
    for i, (key, name) in enumerate(names):
        y = 0.62 - i * 0.085
        row = experiment[key]
        label(fig, 0.55, y, name, size=9, color=INK, weight="bold" if i == 2 else "normal")
        label(fig, 0.76, y, decimal(row["f1_macro"]), size=10, color=TEAL, ha="center")
        label(fig, 0.88, y, f"{row['corretos_d']}/{row['previstos_d']}", size=9, color=INK, ha="center")
    box(fig, 0.06, 0.14, 0.88, 0.22, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.32, "O QUE APRENDEMOS", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.085, 0.275,
            "O limite de 0,25 encontrou 76 empates, mas gerou 228 falsos empates. "
            "Com sinais de equilíbrio e o limite escolhido em 2022, o F1 macro subiu "
            "de 0,331 para 0,370 em 2023; a acurácia caiu de 0,492 para 0,489. "
            "O limite foi escolhido em 2022; 2024 não entrou no experimento. "
            "É um resultado exploratório de uma temporada. O modelo principal continua igual.",
            120, size=9, color=INK)
    return fig


def poisson_page(data: dict, number: int, total: int):
    fig = page("Poisson: probabilidade e decisão", "Modelo de gols", number, total)
    experiment = data["experimento_poisson"]
    baseline = experiment["referencia"]
    poisson = experiment["poisson"]
    box(fig, 0.06, 0.43, 0.42, 0.38)
    label(fig, 0.085, 0.77, "PROBABILIDADES E RESULTADOS", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.12, 0.51, 0.31, 0.21], facecolor=WHITE)
    ax.set_zorder(3)
    values = [baseline["acuracia"], poisson["acuracia"], baseline["f1_macro"], poisson["f1_macro"]]
    x = np.arange(2)
    ax.bar(x - 0.16, [values[0], values[2]], 0.29, color="#B5C7CF", label="Regressão")
    ax.bar(x + 0.16, [values[1], values[3]], 0.29, color=TEAL, label="Poisson")
    ax.set_xticks(x, ["Acurácia", "F1 macro"])
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(labelsize=8, length=0)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    label(fig, 0.10, 0.46, "Log loss: 1,057 → 1,033 (menor é melhor)", size=8, color=MUTED)

    box(fig, 0.52, 0.43, 0.42, 0.38)
    label(fig, 0.55, 0.77, "ACERTOS POR RESULTADO", size=10, color=TEAL, weight="bold")
    for i, (title, key, total_count) in enumerate((("Empate", "d", 98),
                                                     ("Mandante", "h", 178),
                                                     ("Visitante", "a", 104))):
        y = 0.67 - i * 0.105
        label(fig, 0.55, y, title, size=10, color=INK, weight="bold")
        label(fig, 0.76, y, f"{baseline[f'{key}_corretos']}/{total_count}", size=10, color=MUTED)
        label(fig, 0.88, y, f"{poisson[f'{key}_corretos']}/{total_count}", size=10, color=TEAL)
    label(fig, 0.75, 0.72, "Regressão", size=8, color=MUTED)
    label(fig, 0.87, 0.72, "Poisson", size=8, color=MUTED)

    box(fig, 0.06, 0.15, 0.88, 0.23, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.34, "LEITURA DO EXPERIMENTO", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.085, 0.29,
            "Ataque, defesa e mando geram taxas de gols; delas saem probabilidades para H, D e A. "
            "Em 2023 o Poisson teve log loss menor, mas a regra de maior probabilidade não escolheu "
            "nenhum empate. Não há evidência de melhora simultânea em empate, mandante e visitante. "
            "A correção de Dixon-Coles ainda não foi implementada; 2024 não entrou no experimento.",
            119, size=9, color=INK)
    return fig


def round_rule_page(data: dict, number: int, total: int):
    fig = page("Cinco empates por rodada: custo observado", "Regra experimental", number, total)
    experiment = data["experimento_rodadas"]
    reference = experiment["referencia"]
    rule = experiment["regra"]
    box(fig, 0.06, 0.42, 0.54, 0.39)
    label(fig, 0.085, 0.77, "ACERTOS EM 2023 POR RESULTADO", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.12, 0.51, 0.42, 0.21], facecolor=WHITE)
    ax.set_zorder(3)
    x = np.arange(3)
    original = [reference[f"{label}_corretos"] for label in ("d", "h", "a")]
    changed = [rule[f"{label}_corretos"] for label in ("d", "h", "a")]
    ax.bar(x - 0.16, original, 0.30, color="#B5C7CF", label="Poisson")
    ax.bar(x + 0.16, changed, 0.30, color=TEAL, label="Regra")
    ax.set_xticks(x, ["Empate", "Mandante", "Visitante"])
    ax.set_ylim(0, 175)
    ax.set_yticks([0, 50, 100, 150])
    ax.tick_params(labelsize=8, length=0)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    for i, (before, after) in enumerate(zip(original, changed)):
        ax.text(i - 0.16, before + 3, str(before), ha="center", size=7, color=MUTED)
        ax.text(i + 0.16, after + 3, str(after), ha="center", size=7, color=TEAL, fontweight="bold")

    box(fig, 0.64, 0.42, 0.30, 0.39)
    label(fig, 0.67, 0.77, "A REGRA", size=10, color=TEAL, weight="bold")
    label(fig, 0.67, 0.69, f"≤ {experiment['limite_gols']:.2f} gol", size=18, color=NAVY, weight="bold")
    wrapped(fig, 0.67, 0.64, "diferença entre taxas de gols estimadas", 31, size=9, color=MUTED)
    label(fig, 0.67, 0.53, "Até 5 empates", size=15, color=NAVY, weight="bold")
    wrapped(fig, 0.67, 0.49, "por rodada; candidatos ordenados pela menor diferença", 31,
            size=9, color=MUTED)

    box(fig, 0.06, 0.14, 0.88, 0.22, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.32, "CONCLUSÃO", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.085, 0.275,
            "Foram 30 empates corretos entre 101 escolhas de empate; 71 foram falsos. "
            "Os acertos em H caíram de 150 para 126 e em A de 28 para 17. "
            "A igualdade exata das taxas não ocorreu; a rodada 10 teve seis empates reais. "
            "O teto foi respeitado, mas a meta de preservar H e A não foi atingida.",
            118, size=9, color=INK)
    return fig


def long_history_page(data: dict, number: int, total: int):
    fig = page("Mais anos melhoram o modelo?", "Treino temporal", number, total)
    experiment = data["experimento_historico_longo"]
    rows = experiment["metricas_medias_2018_2022"]
    keys = ("recent_3", "recent_5", "recent_10", "all")
    names = ("3 anos", "5 anos", "10 anos", "Todos")

    box(fig, 0.06, 0.42, 0.42, 0.39)
    label(fig, 0.085, 0.77, "F1 MACRO MÉDIO · 2018–2022", size=10, color=TEAL, weight="bold")
    ax = fig.add_axes([0.11, 0.51, 0.33, 0.21], facecolor=WHITE)
    ax.set_zorder(3)
    values = [rows[key]["f1_macro"] for key in keys]
    ax.bar(range(4), values, color=[TEAL, "#8AC3C0", "#A5D0CE", "#B5C7CF"])
    ax.set_xticks(range(4), names)
    ax.set_ylim(0, 0.38)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.tick_params(labelsize=8, length=0)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    for i, value in enumerate(values):
        ax.text(i, value + 0.009, decimal(value), ha="center", size=8, color=INK)
    label(fig, 0.085, 0.46, "Maior F1: treino com 3 temporadas recentes", size=8, color=MUTED)

    box(fig, 0.52, 0.42, 0.42, 0.39)
    label(fig, 0.55, 0.77, "LOG LOSS MÉDIO · 2018–2022", size=10, color=TEAL, weight="bold")
    for i, (key, name) in enumerate(zip(keys, names)):
        y = 0.69 - i * 0.072
        label(fig, 0.55, y, name, size=10, color=INK)
        label(fig, 0.86, y, decimal(rows[key]["log_loss"]), size=10,
              color=TEAL if key == "all" else MUTED, weight="bold" if key == "all" else "normal",
              ha="right")
    label(fig, 0.55, 0.445, "Menor é melhor; todos os anos reduzem esta perda.",
          size=8, color=MUTED)

    box(fig, 0.06, 0.14, 0.88, 0.22, face="#E6F3F2", edge="#C3E3E0")
    label(fig, 0.085, 0.32, "VERIFICAÇÃO RETROSPECTIVA DE 2023", size=10, color=TEAL, weight="bold")
    wrapped(fig, 0.085, 0.275,
            "Com três temporadas recentes: acurácia 49,2%, F1 macro 0,331 e 5 empates corretos. "
            "Com todo o histórico desde 2006: acurácia 46,3%, F1 macro 0,243 e zero empates previstos. "
            "A fonte não tem 2000–2002; 2016 contém 379 jogos. Nenhum método novo substituiu a referência.",
            116, size=9, color=INK)
    return fig


def next_page(data: dict, number: int, total: int):
    fig = page("O que falta e como acompanhar", "Próxima edição", number, total)
    current = data["estado_atual"]
    box(fig, 0.06, 0.40, 0.42, 0.41)
    label(fig, 0.09, 0.765, "ENTREGUE", size=10, color=TEAL, weight="bold")
    for i, text in enumerate(current["entregue"]):
        y = 0.71 - i * 0.061
        label(fig, 0.09, y, "✓", size=12, color=TEAL, weight="bold")
        wrapped(fig, 0.12, y + 0.001, text, 43, size=9, color=INK)
    box(fig, 0.52, 0.40, 0.42, 0.41)
    label(fig, 0.55, 0.765, "PRÓXIMAS ETAPAS", size=10, color=TEAL, weight="bold")
    for i, text in enumerate(current["proximos_passos"]):
        y = 0.71 - i * 0.077
        label(fig, 0.55, y, f"0{i + 1}", size=10, color=TEAL, weight="bold")
        wrapped(fig, 0.59, y + 0.001, text, 41, size=9, color=INK)

    box(fig, 0.06, 0.17, 0.88, 0.20, face=NAVY, edge=NAVY)
    label(fig, 0.09, 0.33, "COMO ESTE REGISTRO CONTINUA", size=10, color=GOLD, weight="bold")
    wrapped(fig, 0.09, 0.29,
            "Ao concluir uma mudança: adicione um marco com data e evidência em docs/evolucao.yaml; "
            "atualize apenas números medidos; execute python docs/build_report.py; "
            "revise o Markdown e o PDF; publique os três arquivos juntos.",
            124, size=10, color=WHITE)
    label(fig, 0.09, 0.205, "FONTES  Git do projeto · CSV processado · relatório local de métricas · previsões de 2024",
          size=8, color="#AFCDD4")
    return fig


def make_pdf(data: dict) -> None:
    milestones = data["marcos"]
    history_chunks = [milestones[i : i + 6] for i in range(0, len(milestones), 6)] or [[]]
    total_pages = 10 + len(history_chunks)
    figures = [cover(data, total_pages)]
    figures.extend(
        history_page(data, chunk, 2 + index, total_pages)
        for index, chunk in enumerate(history_chunks)
    )
    figures.extend(
        [
            data_page(data, total_pages - 8, total_pages),
            model_page(data, total_pages - 7, total_pages),
            validation_page(data, total_pages - 6, total_pages),
            confidence_page(data, total_pages - 5, total_pages),
            draw_experiment_page(data, total_pages - 4, total_pages),
            poisson_page(data, total_pages - 3, total_pages),
            round_rule_page(data, total_pages - 2, total_pages),
            long_history_page(data, total_pages - 1, total_pages),
            next_page(data, total_pages, total_pages),
        ]
    )
    with PdfPages(PDF, metadata={"Title": "Evolução do Brasileirão Predictor",
                                 "Author": "Projeto Brasileirão Predictor",
                                 "Subject": "Histórico e estado do projeto"}) as pdf:
        for figure in figures:
            pdf.savefig(figure, facecolor=figure.get_facecolor())
            plt.close(figure)


def main() -> None:
    data = load_history()
    verify_local_metrics(data)
    verify_local_diagnostics(data)
    verify_local_draw_experiment(data)
    verify_local_poisson_experiment(data)
    verify_local_round_experiment(data)
    verify_local_long_history_experiment(data)
    MARKDOWN.write_text(make_markdown(data), encoding="utf-8")
    make_pdf(data)
    print(f"Markdown: {MARKDOWN}")
    print(f"PDF: {PDF}")


if __name__ == "__main__":
    main()
