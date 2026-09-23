# Brasileirão Predictor

Projeto de ciência de dados para preparar e explorar partidas históricas do
Campeonato Brasileiro. O repositório transforma um CSV público em uma base
padronizada das temporadas de 2020 a 2024, gera features históricas e treina
uma primeira referência de classificação de resultados.

> **Estado atual:** há avaliação retrospectiva e um modelo salvo localmente.
> Ainda não existe uma interface para informar um jogo futuro e obter previsão.

## O que já existe

- Pipeline em Python para carregar, limpar, padronizar e validar os jogos.
- Recorte de cinco temporadas completas (2020--2024), com 1.900 partidas.
- Correção da atribuição da temporada 2020, concluída em janeiro e fevereiro
  de 2021 devido à pandemia.
- Notebook de entendimento dos dados e estatísticas descritivas.
- Notebook de engenharia de features de forma recente, gols e desempenho por
  mando de campo, com uma linha por partida e histórico em ordem cronológica.
- Treino de uma regressão logística e comparação com uma referência que sempre
  prevê a classe mais frequente.
- Testes automatizados para preparação, ordem das features e separação temporal.

## Fluxo de dados

```text
CSV histórico público
        |
        v
src/data/prepare_matches.py
        |
        v
data/processed/matches_2020_2024.csv
        |
        +--> notebooks/02_feature_engineering.ipynb
                    |
                    v
             data/processed/matches_features_2020_2024.csv
                    |
                    v
             src/model/train_baseline.py
                    |
                    +--> artifacts/ (modelo e métricas locais)
                    +--> data/processed/predictions_2024.csv

CSV histórico público --> notebooks/01_data_understanding.ipynb
```

O pipeline espera o arquivo bruto em
`data/raw/campeonato-brasileiro-full.csv` e cria o arquivo processado em
`data/processed/matches_2020_2024.csv`. Ambos os diretórios são ignorados pelo
Git, pois contêm dados obtidos ou gerados localmente.

Durante o processamento, o script:

1. confere se o CSV contém as colunas necessárias;
2. converte as datas no formato `dd/mm/aaaa`;
3. seleciona as temporadas de 2020 a 2024, tratando de forma especial os jogos
   da temporada 2020 realizados até 25/02/2021;
4. renomeia campos para nomes consistentes e calcula `resultado` a partir do
   placar (`H` = vitória do mandante, `D` = empate e `A` = vitória do visitante);
5. valida quantidade de jogos, clubes, rodadas, dados ausentes, resultados e
   duplicatas lógicas antes de salvar o CSV.

O arquivo resultante tem as colunas `id`, `temporada`, `rodada`, `data`,
`mandante`, `visitante`, `gols_mandante`, `gols_visitante`, `arena`,
`estado_mandante`, `estado_visitante` e `resultado`.

## Pré-requisitos

- Git;
- Python 3.12 (versão usada e validada neste repositório);
- `pip` e suporte a ambientes virtuais (`venv`);
- `curl` para executar o comando de download abaixo.

Em Ubuntu/Debian, por exemplo:

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv python3-pip
```

## Como executar

Clone o repositório e prepare um ambiente isolado:

```bash
git clone https://github.com/mvk999/brasileirao-predictor.git
cd brasileirao-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Em sessões futuras, ative novamente o ambiente com:

```bash
source .venv/bin/activate
```

### 1. Baixe a base bruta

O projeto usa o arquivo `campeonato-brasileiro-full.csv` do repositório público
[Brasileirao_Dataset](https://github.com/adaoduque/Brasileirao_Dataset). Baixe-o
para o caminho esperado pelo pipeline:

```bash
mkdir -p data/raw
curl -fL \
  https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv \
  -o data/raw/campeonato-brasileiro-full.csv
```

Também é possível usar uma cópia local do mesmo CSV, desde que ela mantenha as
colunas esperadas pelo script.

### 2. Gere a base processada

Na raiz do repositório, com o ambiente virtual ativo:

```bash
python src/data/prepare_matches.py
```

Ao concluir, o comando informa os caminhos de entrada e saída e gera
`data/processed/matches_2020_2024.csv`. A validação deve confirmar 1.900 jogos:
380 para cada uma das cinco temporadas, 20 clubes por temporada e 38 rodadas
com 10 jogos cada.

### 3. Execute os testes

```bash
python -m pytest
```

Os testes verificam o cálculo do resultado, a exceção da temporada 2020, as
propriedades da base processada, a ordem dos jogos usados nas features e a
separação temporal do treino.

### 4. Abra os notebooks

```bash
python -m jupyter lab
```

Execute as células na ordem em que aparecem:

- `notebooks/01_data_understanding.ipynb` explora a base, sua cobertura e
  estatísticas de partidas e da tabela de 2024.
- `notebooks/02_feature_engineering.ipynb` constrói o histórico dos clubes em
  ordem cronológica, calcula métricas dos cinco jogos anteriores (gerais e por
  mando), reúne mandante e visitante pelo ID da partida e grava
  `data/processed/matches_features_2020_2024.csv`.

A base gerada deve ter 1.900 linhas e 19 colunas: sete identificadores e alvo
(`resultado`), mais seis features para cada time. As médias ficam vazias quando
o clube ainda não tem jogos anteriores naquela temporada ou naquele mando;
isso é esperado e precisa ser tratado na etapa de modelagem. O notebook imprime
`Histórico geral em ordem cronológica conferido.` e `Partidas: 1900 | Features: 12`
ao executar todas as células sem erro.

### 5. Treine e avalie a primeira referência

Depois de gerar a base de features no notebook, execute na raiz do projeto:

```bash
python src/model/train_baseline.py
```

O script usa apenas as 12 métricas históricas como entrada. `resultado` é a
resposta que o modelo aprende a estimar; ID, data, rodada e nomes dos clubes
servem para identificar partidas, mas não entram no modelo. A divisão é feita
por temporada: 2020--2022 para treino (1.140 jogos), 2023 para validação (380)
e 2024 para teste final (380). Depois da validação, o modelo é treinado de novo
com 2020--2023 e avaliado uma vez em 2024.

O tratamento dos valores ausentes aprende a mediana **somente nos dados usados
no treino**. Um indicador informa ao modelo quais médias estavam ausentes, e
as colunas são padronizadas antes da regressão logística. A referência
`DummyClassifier(strategy="prior")` sempre escolhe a classe mais frequente e
usa as frequências observadas no treino como probabilidades.

O comando mostra três métricas para 2023 e 2024: acurácia (fração de acertos),
F1 macro (média do F1 das três classes) e log loss (qualidade das
probabilidades; menor é melhor). Na base local usada neste projeto, o teste de
2024 produziu aproximadamente:

| Método | Acurácia | F1 macro | Log loss |
| --- | ---: | ---: | ---: |
| Referência | 0,474 | 0,214 | 1,057 |
| Regressão logística | 0,482 | 0,327 | 1,035 |

Esses números descrevem apenas essa avaliação histórica. Em 2024, o modelo
previu 323 vitórias do mandante entre 380 jogos; ainda tem dificuldade para
reconhecer empates e vitórias do visitante. As features de cada jogo de 2024
podem usar **resultados de jogos anteriores de 2024**, como aconteceria numa
previsão feita rodada a rodada; o modelo não usa o resultado do jogo avaliado.

O script grava `artifacts/logistic_baseline_2020_2023.joblib`, um JSON com as
métricas em `artifacts/logistic_baseline_metrics.json` e as 380 previsões
retrospectivas em `data/processed/predictions_2024.csv`. Esses arquivos são
gerados localmente e ignorados pelo Git.

## Estrutura do repositório

```text
src/data/prepare_matches.py    Pipeline e validações da base de partidas
src/model/train_baseline.py     Treino e avaliação temporal do primeiro modelo
tests/                         Testes do pipeline
notebooks/                     Exploração e engenharia de features
data/raw/                      CSV baixado localmente (não versionado)
data/processed/                CSV gerado pelo pipeline (não versionado)
.env.example                   Exemplo de variáveis para integrações futuras
```

## Configuração e segurança

Nenhuma variável de ambiente é necessária para executar o pipeline, os testes,
os notebooks ou o treino atual. O arquivo `.env.example` é apenas uma referência
para possíveis integrações futuras (banco de dados, API de futebol e LLM). Caso
crie um `.env`, mantenha segredos fora do Git: esse arquivo já é ignorado pelo
repositório.

## Limitações e próximos passos

Este repositório contém uma primeira avaliação retrospectiva, não um produto
final. As próximas etapas incluem entender os erros por classe, melhorar as
features e a avaliação sem ajustar o modelo ao teste de 2024, atualizar os dados
para temporadas posteriores e criar um fluxo que gere features para jogos ainda
não disputados. Só depois faz sentido expor previsões por uma API ou interface.
Os resultados atuais não constituem recomendação de aposta.

## Solução de problemas

- **`Dataset bruto não encontrado`**: baixe o CSV e confirme o caminho
  `data/raw/campeonato-brasileiro-full.csv`.
- **Erro sobre colunas necessárias**: use o CSV indicado acima ou adapte a
  origem para conter os campos esperados pelo pipeline.
- **Erro de validação de quantidade de partidas**: a fonte pode ter sido
  alterada ou o arquivo pode estar incompleto; obtenha novamente a versão
  compatível com o recorte 2020--2024.
- **Notebook não encontra o CSV processado**: execute antes
  `python src/data/prepare_matches.py` a partir da raiz do projeto.
