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


def make_markdown(data: dict) -> str:
    model = data["modelo"]
    metrics = model["metricas"]
    total_games = f"{data['dados']['total_jogos']:,}".replace(",", ".")
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
    lines += ["", "## O que existe hoje", ""]
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
        "   com o commit que comprova o que foi feito.",
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
    fig = page("Como o projeto chegou até aqui", "Trajetória verificada no Git", number, total)
    fig.lines.append(mpl.lines.Line2D([0.105, 0.105], [0.17, 0.79], transform=fig.transFigure,
                                       color=TEAL, lw=2.2))
    for index, item in enumerate(items):
        y = 0.79 - index * 0.11
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
            "CSV preparado e validado; 12 métricas históricas por partida; primeiro modelo avaliado em 2024.",
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


def next_page(data: dict, number: int, total: int):
    fig = page("O que falta e como acompanhar", "Próxima edição", number, total)
    current = data["estado_atual"]
    box(fig, 0.06, 0.43, 0.42, 0.38)
    label(fig, 0.09, 0.765, "ENTREGUE", size=10, color=TEAL, weight="bold")
    for i, text in enumerate(current["entregue"]):
        y = 0.71 - i * 0.077
        label(fig, 0.09, y, "✓", size=12, color=TEAL, weight="bold")
        wrapped(fig, 0.12, y + 0.001, text, 43, size=9, color=INK)
    box(fig, 0.52, 0.43, 0.42, 0.38)
    label(fig, 0.55, 0.765, "PRÓXIMAS ETAPAS", size=10, color=TEAL, weight="bold")
    for i, text in enumerate(current["proximos_passos"]):
        y = 0.71 - i * 0.077
        label(fig, 0.55, y, f"0{i + 1}", size=10, color=TEAL, weight="bold")
        wrapped(fig, 0.59, y + 0.001, text, 41, size=9, color=INK)

    box(fig, 0.06, 0.17, 0.88, 0.20, face=NAVY, edge=NAVY)
    label(fig, 0.09, 0.33, "COMO ESTE REGISTRO CONTINUA", size=10, color=GOLD, weight="bold")
    wrapped(fig, 0.09, 0.29,
            "Ao concluir uma mudança: adicione um marco com data e commit em docs/evolucao.yaml; "
            "atualize apenas números medidos; execute python docs/build_report.py; "
            "revise o Markdown e o PDF; publique os três arquivos juntos.",
            124, size=10, color=WHITE)
    label(fig, 0.09, 0.205, "FONTES  Git do projeto · CSV processado · relatório local de métricas · previsões de 2024",
          size=8, color="#AFCDD4")
    return fig


def make_pdf(data: dict) -> None:
    milestones = data["marcos"]
    history_chunks = [milestones[i : i + 6] for i in range(0, len(milestones), 6)] or [[]]
    total_pages = 4 + len(history_chunks)
    figures = [cover(data, total_pages)]
    figures.extend(
        history_page(data, chunk, 2 + index, total_pages)
        for index, chunk in enumerate(history_chunks)
    )
    figures.extend(
        [
            data_page(data, total_pages - 2, total_pages),
            model_page(data, total_pages - 1, total_pages),
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
    MARKDOWN.write_text(make_markdown(data), encoding="utf-8")
    make_pdf(data)
    print(f"Markdown: {MARKDOWN}")
    print(f"PDF: {PDF}")


if __name__ == "__main__":
    main()
