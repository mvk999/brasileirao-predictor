# Brasileirão Predictor

Projeto experimental de ciência de dados que transforma resultados históricos do Campeonato Brasileiro em atributos anteriores a cada partida e avalia modelos probabilísticos para vitória do mandante, empate e vitória do visitante. O foco está no processo: preparar dados confiáveis, evitar vazamento temporal e medir onde as previsões funcionam ou falham.

**Estado atual:** o projeto produz avaliações retrospectivas e salva um modelo local. Ainda não recebe uma partida futura para gerar uma previsão em tempo real.

[Resultados](#results) · [Como funciona](#how-it-works) · [Executar localmente](#running-locally) · [Histórico e relatório visual](docs/EVOLUCAO.md) · [PDF](docs/evolucao-do-projeto.pdf)

## Motivation

Resultados de futebol são difíceis de estimar: o desempenho recente contém informação, mas uma partida também depende de acontecimentos que as médias históricas não descrevem. Isso torna o problema útil para estudar a diferença entre **encontrar padrões** e **generalizar para jogos futuros**.

O projeto investiga perguntas concretas:

- Quanto uma referência simples, que sempre escolhe o resultado mais comum, já consegue acertar?
- Estatísticas calculadas antes do jogo melhoram essa referência?
- O modelo reconhece empates e vitórias do visitante ou concentra as escolhas no mandante?
- Probabilidades melhores levam necessariamente a mais acertos em cada classe?

A resposta é tratada como estimativa com incerteza, não como certeza sobre o placar.

## Results

O recorte validado contém **1.900 partidas**, 380 por temporada de 2020 a 2024. A primeira referência é uma regressão logística treinada com 12 atributos históricos. `H` significa vitória do mandante, `D` empate e `A` vitória do visitante.

| Período | Método | Acurácia | F1 macro | Log loss |
| --- | --- | ---: | ---: | ---: |
| Validação 2023 | Classe mais frequente | 46,8% | 0,213 | 1,062 |
| Validação 2023 | Regressão logística | 49,2% | 0,331 | 1,057 |
| Avaliação retrospectiva 2024 | Classe mais frequente | 47,4% | 0,214 | 1,057 |
| Avaliação retrospectiva 2024 | Regressão logística | 48,2% | 0,327 | 1,035 |

A tabela mostra um avanço pequeno em acurácia. A análise por classe revela uma limitação maior: em 2023 houve **98 empates**, mas a regressão escolheu empate em apenas **7 jogos** e acertou **5**.

Exemplo retrospectivo da validação: em **Goiás × Grêmio, 30/07/2023**, o modelo atribuiu 31,4% a `A`, 38,9% a `D` e 29,7% a `H`. Escolheu empate, que foi o resultado real. É um dos cinco empates reconhecidos, não uma previsão feita antes da partida em produção; o registro pode ser reproduzido em `data/processed/predictions_2023_validation.csv`.

### Experimentos com empates

Um limite de decisão escolhido em 2022 aumentou, em 2023, os empates corretos de 5 para 10, com 21 previsões de empate. Três sinais adicionais de equilíbrio elevaram esse número para 18 em 42 previsões, mas a acurácia caiu de 49,2% para 48,9%. São resultados exploratórios; o modelo principal não foi substituído. O experimento completo está em [`experiment_draws.py`](src/model/experiment_draws.py).

Também foi avaliado um modelo de **gols por Poisson independente**. Ele estima forças de ataque e vulnerabilidade defensiva a partir de partidas anteriores e converte taxas de gols em probabilidades de `H`, `D` e `A`. Em 2023, seu log loss foi **1,033**, ante **1,057** da regressão. Porém, ao escolher a classe de maior probabilidade, ele **não previu nenhum empate** e sua acurácia foi **46,8%**. O experimento está em [`experiment_poisson.py`](src/model/experiment_poisson.py). A correção de Dixon–Coles ainda não foi implementada.

Uma nova regra experimental escolhe empate quando as **taxas de gols estimadas** dos times diferem em até 0,20 gol e limita as escolhas a cinco por rodada. O limite foi escolhido em 2022. Em 2023, encontrou **30 empates em 101 previsões de empate**, mas os acertos de `H` caíram de 150 para 126 e os de `A` de 28 para 17, na comparação com o Poisson sem a regra. A acurácia caiu de 46,8% para 45,5%. A regra não entrou no modelo principal; veja [`experiment_round_draws.py`](src/model/experiment_round_draws.py).

Os gráficos, matrizes de confusão e a trajetória do projeto estão no [relatório de evolução](docs/EVOLUCAO.md) e no [PDF](docs/evolucao-do-projeto.pdf). As previsões jogo a jogo são arquivos gerados localmente, em `data/processed/`.

## How it works

```text
CSV público de partidas
        ↓
Validação e padronização ──→ data/processed/matches_2020_2024.csv
        ↓
Atributos de jogos anteriores ──→ data/processed/matches_features_2020_2024.csv
        ↓
Treino por temporadas ──→ probabilidades de H / D / A
        ↓
Avaliação retrospectiva ──→ métricas, diagnósticos e relatório
```

1. **Coleta:** o CSV bruto é obtido manualmente da fonte pública indicada abaixo.
2. **Preparação:** o script confere colunas, datas, clubes, rodadas, placares e duplicatas. O resultado `H/D/A` é calculado a partir dos gols.
3. **Engenharia de atributos:** o notebook organiza o histórico de cada clube em ordem cronológica e calcula médias dos cinco jogos anteriores, gerais e por mando. Cada jogo recebe seis atributos do mandante e seis do visitante.
4. **Treino:** um pipeline ajusta valores ausentes e escala apenas com os dados do período de treino; depois aprende uma regressão logística.
5. **Avaliação:** as previsões são comparadas com resultados de temporadas posteriores. Scripts separados investigam limites de empate e um modelo de gols.

Os scripts atuais recebem **arquivos CSV**, não requisições de uma API. Suas saídas são CSVs de previsões, JSONs de métricas, um modelo `joblib` e a documentação gerada.

## Dataset

A fonte é o arquivo [`campeonato-brasileiro-full.csv` do Brasileirao_Dataset](https://github.com/adaoduque/Brasileirao_Dataset). O pipeline usa ID, rodada, data, mandante, visitante, placares, arena e estados dos clubes. A partir do placar, calcula o alvo `resultado`.

O recorte usa as temporadas **2020 a 2024**. Jogos de janeiro e fevereiro de 2021 que concluíram o Brasileirão 2020 são atribuídos à temporada 2020, conforme a regra explícita em [`prepare_matches.py`](src/data/prepare_matches.py). A validação exige 380 jogos, 20 clubes e 38 rodadas por temporada.

Os dados brutos e processados ficam em `data/raw/` e `data/processed/`, ignorados pelo Git. Médias históricas dos primeiros jogos de um clube podem estar vazias; isso expressa ausência de histórico, não um resultado igual a zero. O pipeline do modelo aprende a mediana de substituição apenas no conjunto de treino.

## Machine Learning Approach

**Referência principal.** [`train_baseline.py`](src/model/train_baseline.py) compara uma `DummyClassifier(strategy="prior")` com uma regressão logística multiclasse. A regressão foi escolhida como ponto de partida simples e interpretável para estabelecer uma medida de referência. O pipeline usa imputação pela mediana com indicador de ausência, padronização e `LogisticRegression`.

**Separação temporal.** O treino inicial usa 2020–2022; 2023 serve para validação. Depois, o modelo é ajustado novamente com 2020–2023 e avaliado retrospectivamente em 2024. Para cada partida avaliada, os atributos usam apenas jogos anteriores da mesma temporada. O teste de 2024 já foi examinado neste projeto e não deve ser tratado como um conjunto inteiramente novo para futuras escolhas de modelo.

**Experimentos.** O limite de empate é escolhido em 2022 e comparado em 2023. O modelo de gols segue a ideia de forças de ataque e defesa estudada por [Maher (1982)](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x): usa resultados anteriores à data de cada partida, escolhe a intensidade de regularização em 2022 e compara seu desempenho em 2023. Sua versão atual usa Poissons independentes; ela não implementa a correção de placares baixos de [Dixon e Coles (1997)](https://doi.org/10.1111/1467-9876.00065). Nenhum desses experimentos substitui o modelo principal ou constitui previsão de uma partida futura.

Na regra por rodada, todas as dez partidas usam um retrato do histórico disponível **antes do primeiro jogo da rodada**. Isso impede que o resultado de um jogo da rodada influencie a decisão para outro. As taxas calculadas com placares passados não são **xG de finalizações**: xG, no sentido usual, estima a chance de cada finalização virar gol a partir de suas características, como explica a [StatsBomb](https://statsbomb.com/soccer-metrics/expected-goals-xg-explained/). Esta base não contém esses eventos.

## Technical Decisions

| Decisão | Motivo | Consequência ou limite |
| --- | --- | --- |
| Validar temporadas completas | Detectar recortes incompletos e inconsistências estruturais | O pipeline espera exatamente o recorte 2020–2024 |
| Ordenar partidas antes das médias históricas | Impedir que o resultado do próprio jogo entre nos atributos | Primeiros jogos têm médias ausentes |
| Separar treino e avaliação por temporada | Respeitar a ordem em que os resultados seriam conhecidos | Há poucas temporadas para confirmar generalização |
| Usar uma referência simples antes de modelos mais complexos | Medir o ganho real de cada abordagem | A regressão ainda tem baixo desempenho em empates |
| Registrar métricas por classe e log loss | Expor erros que a acurácia geral esconde | Melhor probabilidade não garante melhor decisão por classe |
| Testar o teto de cinco empates como hipótese | Medir a regra sugerida sem consultar o resultado real ao escolher empates | A rodada 10 de 2023 teve seis empates reais; o teto impede reconhecer todos |
| Manter experimentos em scripts separados | Preservar uma referência reproduzível enquanto novas hipóteses são avaliadas | Resultados experimentais ainda exigem confirmação |

## Technologies

| Área | Tecnologias usadas | Papel no projeto |
| --- | --- | --- |
| Linguagem | Python 3.12 | Scripts de preparação, modelagem e relatório |
| Dados | Pandas, NumPy | Tabelas, transformações e cálculos numéricos |
| Modelagem | scikit-learn, SciPy | Regressão logística, métricas e distribuição de gols |
| Exploração | JupyterLab | Inspeção e engenharia de atributos em notebooks |
| Visualização e documentação | Matplotlib, PyYAML | Gráficos, PDF e histórico editável |
| Verificação | pytest | Regras de preparação, ordem temporal e funções experimentais |

## Project Structure

```text
brasileirao-predictor/
├── src/
│   ├── data/prepare_matches.py       # limpeza e validação do CSV
│   └── model/
│       ├── train_baseline.py         # referência, treino e avaliação
│       ├── experiment_draws.py       # limites e sinais de equilíbrio
│       ├── experiment_poisson.py     # modelo experimental de gols
│       └── experiment_round_draws.py # regra de empate por rodada
├── notebooks/
│   ├── 01_data_understanding.ipynb   # exploração da fonte
│   └── 02_feature_engineering.ipynb  # atributos históricos por partida
├── tests/                            # verificações automatizadas
├── docs/                             # histórico em YAML, Markdown e PDF
├── data/raw/                         # CSV baixado; ignorado pelo Git
├── data/processed/                   # CSVs gerados; ignorados pelo Git
├── artifacts/                        # modelo e métricas locais; ignorados
├── requirements.txt
└── README.md
```

## Running locally

Requisitos: Git, Python 3.12, `venv`, `pip` e `curl`. Na raiz do projeto:

```bash
git clone https://github.com/mvk999/brasileirao-predictor.git
cd brasileirao-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/raw
curl -fL https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv -o data/raw/campeonato-brasileiro-full.csv
python src/data/prepare_matches.py
python -m jupyter lab
```

No JupyterLab, execute `notebooks/02_feature_engineering.ipynb` em ordem para gerar a base de atributos. O notebook de exploração `01_data_understanding.ipynb` é opcional para o treino. Depois:

```bash
python src/model/train_baseline.py
python -m src.model.experiment_draws
python -m src.model.experiment_poisson
python -m src.model.experiment_round_draws
python -m pytest -q
python docs/build_report.py
```

A preparação deve informar **1.900 partidas**; o notebook de atributos termina com `Partidas: 1900 | Features: 12`. O treino imprime métricas para 2023 e 2024; os experimentos imprimem comparações de 2023. O experimento por rodada mostra `Limite escolhido em 2022: 0.20 gol; teto: 5 empates/rodada`. O `pytest` informa o total de testes aprovados. O gerador de documentação escreve `docs/EVOLUCAO.md` e `docs/evolucao-do-projeto.pdf`. Nenhuma variável de ambiente é exigida por esse fluxo; `.env.example` é apenas uma referência para possíveis integrações futuras.

## Evaluation

- **Acurácia:** fração de jogos com classe `H`, `D` ou `A` correta.
- **F1 macro:** média do F1 das três classes, para que uma classe pouco reconhecida afete a avaliação.
- **Precisão, revocação e F1 por classe:** mostram, por exemplo, quantos empates previstos foram corretos e quantos empates reais foram encontrados.
- **Log loss:** mede a qualidade das três probabilidades previstas; menor é melhor.
- **Matriz de confusão:** mostra quais resultados reais foram trocados por quais previsões.

O diagnóstico de 2023 e os experimentos mostram por que é necessário olhar além da acurácia: prever quase sempre vitória do mandante pode produzir uma taxa global razoável e ainda deixar a maioria dos empates sem reconhecimento. A comparação de novas ideias usa temporadas posteriores às de treino e deve ser repetida com dados verdadeiramente novos antes de afirmar uma melhora estável.

## Limitations

- O desempenho atual é limitado: a regressão acertou 48,2% das partidas de 2024, apenas três jogos a mais que a referência simples.
- Em 2023, a regressão reconheceu 5 dos 98 empates; o Poisson independente, pela regra de maior probabilidade, não previu nenhum.
- A regra de taxas próximas reconheceu mais empates, mas reduziu os acertos de `H` e `A`; o teto de cinco não vem de uma propriedade estatística demonstrada.
- Os atributos cobrem forma recente e gols, mas não capturam escalações, lesões, contexto tático ou outras condições da partida.
- Há somente cinco temporadas no recorte atual; mudanças de elenco e clubes promovidos dificultam extrapolar forças entre anos.
- As análises de 2023 e 2024 já influenciaram a investigação. Uma nova temporada é necessária para uma confirmação mais independente.
- O projeto ainda não possui rotina de atualização automática, entrada para jogos não disputados, API ou interface de usuário.
- Os resultados são estudos retrospectivos e não constituem recomendação de aposta.

## Future Improvements

1. Obter e validar temporadas posteriores a 2024 para uma avaliação mais independente.
2. Investigar calibração das probabilidades, regras de decisão e a correção de placares baixos de Dixon–Coles, sem assumir que melhorarão empates.
3. Testar atributos anteriores ao jogo que representem força dos times e contexto de forma verificável.
4. Automatizar a atualização da fonte e criar um fluxo seguro para montar atributos de partidas futuras.
5. Considerar API ou interface visual após validar esse fluxo e os resultados fora da amostra.

## What I learned

Ao construir este projeto, aprendi a tratar uma data de calendário e uma temporada esportiva como conceitos diferentes; a proteger a ordem cronológica na engenharia de atributos; a interpretar valores ausentes no início de uma série; e a separar treino e avaliação de acordo com o momento em que cada resultado se tornou conhecido.

A comparação por classe mostrou que **acurácia sozinha é insuficiente**. Os experimentos com empates e gols reforçaram outra distinção: um modelo pode atribuir probabilidades melhores aos resultados e ainda tomar decisões piores quando precisa escolher uma única classe. Registrar resultados negativos e preservar a referência torna a evolução do projeto verificável.

## Author

**Marcos Vinícius Pereira** — Estudante de Ciência da Computação interessado em desenvolvimento de software, backend, dados e construção de soluções utilizando tecnologia.

[GitHub](https://github.com/mvk999) · [LinkedIn](https://www.linkedin.com/in/mvpereira2006/)
