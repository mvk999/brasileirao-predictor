# Brasileirão Predictor

Projeto de ciência de dados para preparar e explorar partidas históricas do
Campeonato Brasileiro. Nesta fase, o repositório transforma um CSV público em
uma base padronizada das temporadas de 2020 a 2024 e documenta análises e
features candidatas em notebooks.

> **Estado atual:** o projeto ainda não treina um modelo nem disponibiliza uma
> interface ou comando para prever partidas. Portanto, ele não gera previsões
> por enquanto.

## O que já existe

- Pipeline em Python para carregar, limpar, padronizar e validar os jogos.
- Recorte de cinco temporadas completas (2020--2024), com 1.900 partidas.
- Correção da atribuição da temporada 2020, concluída em janeiro e fevereiro
  de 2021 devido à pandemia.
- Notebook de entendimento dos dados e estatísticas descritivas.
- Notebook exploratório de engenharia de features de forma recente, gols e
  desempenho por mando de campo.
- Testes automatizados para as regras centrais de preparação dos dados.

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
        +--> notebooks/01_data_understanding.ipynb
        +--> notebooks/02_feature_engineering.ipynb
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

Os testes verificam o cálculo do resultado, a exceção da temporada 2020 e as
propriedades esperadas da base processada.

### 4. Abra os notebooks

```bash
python -m jupyter lab
```

Execute as células na ordem em que aparecem:

- `notebooks/01_data_understanding.ipynb` explora a base, sua cobertura e
  estatísticas de partidas e da tabela de 2024.
- `notebooks/02_feature_engineering.ipynb` constrói, em memória, o histórico
  dos clubes e métricas de últimos cinco jogos, inclusive recortes por mando.
  Ele depende do CSV processado e não grava uma base de features no disco.

## Estrutura do repositório

```text
src/data/prepare_matches.py    Pipeline e validações da base de partidas
tests/                         Testes do pipeline
notebooks/                     Exploração e engenharia de features
data/raw/                      CSV baixado localmente (não versionado)
data/processed/                CSV gerado pelo pipeline (não versionado)
.env.example                   Exemplo de variáveis para integrações futuras
```

## Configuração e segurança

Nenhuma variável de ambiente é necessária para executar o pipeline, os testes
ou os notebooks atuais. O arquivo `.env.example` é apenas uma referência para
possíveis integrações futuras (banco de dados, API de futebol e LLM). Caso crie
um `.env`, mantenha segredos fora do Git: esse arquivo já é ignorado pelo
repositório.

## Limitações e próximos passos

Este repositório é a fundação de dados de um preditor, não o produto final. As
etapas que ainda faltam incluem consolidar uma base de features reutilizável,
treinar e avaliar modelos, versionar artefatos e expor previsões por uma API ou
interface. As métricas dos notebooks são exploratórias e não constituem, por si
só, previsões ou recomendação de aposta.

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
