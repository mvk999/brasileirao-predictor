# Brasileirão Predictor

Aplicação educacional para importar dados históricos do Campeonato Brasileiro,
treinar modelos estatísticos e de machine learning e gerar previsões
probabilísticas para partidas futuras.

## Estado do projeto

Projeto em desenvolvimento.

## Objetivos

- Importar partidas e resultados históricos.
- Calcular estatísticas dos clubes.
- Treinar e versionar modelos preditivos.
- Gerar previsões probabilísticas.
- Avaliar previsões após os jogos.
- Utilizar LLM apenas para explicar resultados calculados pelo modelo.


  ## Como executar o projeto

  ### Pré-requisitos

  Em Ubuntu/Debian, instale Python 3, o suporte a ambientes virtuais (`venv`),
  `pip`, Git e `curl`:

  ```bash
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip git curl

  O requirements.txt instala as bibliotecas de análise, testes e o JupyterLab.

  ### Preparar o ambiente

  Depois de clonar o repositório, entre na pasta do projeto e crie um ambiente
  virtual. Isso mantém as dependências do projeto isoladas do Python do sistema:

  cd brasileirao-predictor
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  O prefixo (.venv) no terminal indica que o ambiente está ativo. Em novas
  sessões, ative-o novamente com source .venv/bin/activate.

  ### Obter os dados brutos

  O dataset não é versionado neste repositório. Baixe o arquivo de partidas da
  fonte pública e salve-o no caminho esperado pelo pipeline:

  mkdir -p data/raw
  curl -fL \
    https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv \
    -o data/raw/campeonato-brasileiro-full.csv

  Os diretórios data/raw/ e data/processed/ são ignorados pelo Git.

  ### Preparar o dataset

  Com o ambiente virtual ativo, execute o pipeline a partir da raiz do projeto:

  python src/data/prepare_matches.py

  O comando valida e prepara as partidas das temporadas de 2020 a 2024, gerando
  data/processed/matches_2020_2024.csv com 1.900 partidas.

  Para executar os testes automatizados:

  python -m pytest

  ### Abrir os notebooks

  Inicie o JupyterLab com o ambiente virtual ativo:

  python -m jupyter lab

  Abra os notebooks pela interface exibida no navegador. O notebook de
  compreensão dos dados está em notebooks/01_data_understanding.ipynb.

## Regras de desenvolvimento

Leia o arquivo [AGENTS.md](AGENTS.md) antes de realizar qualquer alteração.
