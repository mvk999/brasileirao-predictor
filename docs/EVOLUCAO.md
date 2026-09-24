# Evolução do Brasileirão Predictor

**Edição 5 · Atualizado em 24/09/2026**

Este é o registro cronológico do projeto. `evolucao.yaml` contém os fatos
editáveis; `EVOLUCAO.md` e `evolucao-do-projeto.pdf` são gerados a partir dele.

## Como começou

O repositório começou como um projeto educacional. O README inicial descrevia o plano de importar partidas históricas, calcular estatísticas, treinar modelos e, futuramente, gerar previsões probabilísticas. Esses itens eram objetivos, não funcionalidades prontas.

Evidência inicial: commit `fee7978` de 25/07/2026.

## Linha do tempo

| Data | Marco | O que passou a existir | Evidência |
| --- | --- | --- | --- |
| 25/07/2026 | Ideia e estrutura inicial | README com objetivos, configurações básicas e organização do repositório. | `fee7978` |
| 27/08/2026 | Inspeção do CSV histórico | Primeiro notebook de entendimento, verificação de datas, resultados e cobertura. | `aece4c2` |
| 05/09/2026 | Pipeline de preparação | Script, testes e recorte validado de 2020 a 2024; exploração de features em notebook. | `771ac26, 6337e36` |
| 06/09/2026 | Estado documentado | README passou a separar claramente exploração de uma previsão funcional. | `96f23d7` |
| 23/09/2026 | Features por partida | Ordem cronológica corrigida e base com seis métricas para cada time. | `ccaabcb` |
| 23/09/2026 | Primeiro modelo avaliado | Regressão logística e referência simples, com validação em 2023 e teste em 2024. | `c544fac` |
| 23/09/2026 | Erros de 2023 analisados | Matriz de confusão, métricas por classe, faixas de confiança e exemplos de erros. | `3066650` |
| 23/09/2026 | Experimento com empates | Limites de decisão e três sinais de equilíbrio comparados com validação temporal. | `experiment_draws.py` |
| 23/09/2026 | Modelo de gols por Poisson | Forças de ataque e defesa estimadas sem olhar jogos futuros; comparação com o modelo de classes. | `experiment_poisson.py` |
| 24/09/2026 | Empates por rodada | Taxas de gols próximas e teto de cinco empates testados sem consultar resultados futuros. | `experiment_round_draws.py` |

## Estado atual dos dados

O recorte contém **1.900 jogos** de 2020 a 2024, com **12 features por partida** em uma tabela de 19 colunas.

| Temporada | Partidas |
| --- | ---: |
| 2020 | 380 |
| 2021 | 380 |
| 2022 | 380 |
| 2023 | 380 |
| 2024 | 380 |

O alvo é `H = mandante; D = empate; A = visitante`. O histórico de cada clube considera somente jogos anteriores na mesma temporada. Médias dos primeiros jogos ficam vazias por falta de histórico.

```text
CSV histórico → preparação → exploração → features por partida
             → treino temporal → avaliação retrospectiva de 2024
```

## Primeiro modelo e avaliação

**Modelo:** Regressão logística multiclasse. **Entrada:** 12 métricas históricas de mandante e visitante. **Preparo:** Mediana aprendida no treino, indicadores de ausência e padronização.

**Separação:** treino 2020–2022: 1.140 jogos; validação 2023: 380 jogos; teste 2024: 380 jogos. O modelo foi treinado novamente com 2020–2023 antes do teste final de 2024.

**Referência de comparação:** Classe mais frequente no treino (DummyClassifier com prior).

| Período | Método | Acurácia | F1 macro | Log loss |
| --- | --- | ---: | ---: | ---: |
| Validação 2023 | Referência | 46,8% | 0,213 | 1,062 |
| Validação 2023 | Regressão logística | 49,2% | 0,331 | 1,057 |
| Teste 2024 | Referência | 47,4% | 0,214 | 1,057 |
| Teste 2024 | Regressão logística | 48,2% | 0,327 | 1,035 |

O teste de 2024 é retrospectivo. As features de cada partida podem usar
resultados **anteriores** da mesma temporada; não usam o resultado da
partida avaliada. O desempenho ainda é limitado: a regressão previu
**323 vitórias do mandante** em 380 jogos,
mas só **17 empates**. A matriz abaixo usa
linhas para o resultado real e colunas para a previsão.

| Real / Previsto | A | D | H |
| --- | ---: | ---: | ---: |
| A | 13 | 4 | 82 |
| D | 12 | 9 | 80 |
| H | 15 | 4 | 161 |

## Diagnóstico dos erros na validação de 2023

Os parâmetros do modelo foram treinados somente com 2020–2022. As features de cada jogo de 2023 usam apenas resultados de partidas anteriores.

A matriz usa **linhas para o resultado real** e **colunas para a previsão**:

| Real / Previsto | A | D | H |
| --- | ---: | ---: | ---: |
| A | 19 | 2 | 83 |
| D | 14 | 5 | 79 |
| H | 15 | 0 | 163 |

| Classe | Reais | Previstos | Acertos | Precisão | Revocação | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 104 | 48 | 19 | 39,6% | 18,3% | 0,250 |
| D | 98 | 7 | 5 | 71,4% | 5,1% | 0,095 |
| H | 178 | 325 | 163 | 50,2% | 91,6% | 0,648 |

O modelo reconheceu **5 dos 98 empates** (5,1% de revocação), pois só escolheu D em 7 jogos. Em contraste, acertou **163 das 178 vitórias do mandante** (91,6% de revocação), mas escolheu H em 325 jogos. Isso mostra concentração das previsões em H; não demonstra, por si só, a causa do comportamento.

### Confiança e acerto observado

A confiança é a maior das três probabilidades previstas. Cada faixa
compara a confiança média com a proporção de acertos nas partidas nela
incluídas. Faixas pequenas exigem cautela na interpretação.

| Confiança prevista | Jogos | Confiança média | Acerto observado |
| --- | ---: | ---: | ---: |
| 0,0%–40,0% | 78 | 37,6% | 39,7% |
| 40,0%–50,0% | 172 | 44,8% | 52,3% |
| 50,0%–60,0% | 101 | 54,5% | 48,5% |
| 60,0%–70,0% | 27 | 64,1% | 55,6% |
| 70,0%–100,0% | 2 | 74,5% | 100,0% |

### Cinco erros com maior confiança na classe prevista

Esses exemplos ajudam a inspecionar o comportamento do modelo; não
explicam isoladamente por que ele errou.

| Data | Partida | Real | Previsto | Confiança |
| --- | --- | --- | --- | ---: |
| 08/10/2023 | Atletico-MG × Coritiba (ID 8283) | A | H | 67,6% |
| 29/04/2023 | Coritiba × Sao Paulo (ID 8047) | D | A | 67,3% |
| 23/04/2023 | Vasco × Palmeiras (ID 8042) | D | H | 67,1% |
| 20/05/2023 | America-MG × Fortaleza (ID 8090) | H | A | 66,4% |
| 23/04/2023 | Santos × Atletico-MG (ID 8041) | D | H | 66,0% |

## Experimento com empates

Treino 2020–2021 e escolha do limite em 2022 pelo maior F1 macro; retreino 2020–2022 e avaliação em 2023. A temporada 2024 não foi usada. O limite é escolhido separadamente para cada conjunto de features.

A regra ajustada escolhe empate quando sua probabilidade atinge o
limite; caso contrário, escolhe entre mandante e visitante. A tabela
mostra o desempenho em 2023. A mudança de limite não altera as
probabilidades, então o log loss da mesma linha de modelo não muda.

| Método | Limite D | Acurácia | F1 macro | Log loss | D previstos | D corretos | Precisão D | Revocação D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Referência original | maior probabilidade | 49,2% | 0,331 | 1,057 | 7 | 5 | 71,4% | 5,1% |
| Referência com limite | 0.35 | 49,7% | 0,357 | 1,057 | 21 | 10 | 47,6% | 10,2% |
| Equilíbrio com limite | 0.34 | 48,9% | 0,370 | 1,055 | 42 | 18 | 42,9% | 18,4% |

**Novos sinais testados:** Diferença absoluta de pontos por jogo, diferença absoluta de gols estimados e interação entre essa diferença e o total estimado de gols. Tudo deriva de médias de partidas anteriores à partida avaliada.

### Troca entre empates encontrados e falsos empates

Os limites abaixo são apenas ilustrativos em 2023, com o modelo de 12 features.

| Limite | Empates previstos | Empates corretos | Falsos empates | Acurácia |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 304 | 76 | 228 | 29,5% |
| 0.30 | 126 | 36 | 90 | 43,2% |
| 0.35 | 21 | 10 | 11 | 49,7% |

**Conclusão:** O limite de 0,35 da referência elevou o F1 macro de 0,331 para 0,357, com 10 dos 98 empates reconhecidos. As features adicionais com limite de 0,34 elevaram o F1 macro a 0,370 e reconheceram 18 empates, mas a acurácia ficou em 0,489 ante 0,492 da referência. O ganho é exploratório: uma única temporada de validação e 2023 já havia sido inspecionada. O modelo usado pelo projeto não foi substituído.

## Experimento com gols por Poisson

Para cada jogo, são usados somente resultados com data anterior. Ataque e vulnerabilidade defensiva de cada clube usam gols marcados/sofridos em relação à média da liga; dados de temporadas mais antigas recebem peso menor. A intensidade de encolhimento foi escolhida em 2022 pelo menor log loss; 2023 foi usado para comparação. 2024 não foi usado.

Poisson independente para gols da casa e de fora, com vantagem de mando embutida nas médias de gols da liga. A distribuição Skellam fornece as probabilidades de A, D e H. Não foi aplicada a correção de Dixon-Coles.

| Método em 2023 | Acurácia | F1 macro | Log loss | Empates corretos / previstos | H corretos / reais | A corretos / reais |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regressão logística | 49,2% | 0,331 | 1,057 | 5/7 | 163/178 | 19/104 |
| Poisson independente | 46,8% | 0,311 | 1,033 | 0/0 | 150/178 | 28/104 |

**Conclusão:** A probabilidade média de empate do Poisson em 2023 ficou próxima da frequência real, mas empate nunca foi a classe mais provável. O log loss melhorou; acurácia, F1 macro e acertos de mandante pioraram. O modelo principal não foi substituído. A melhoria de empate sem perda de H e A ainda não foi demonstrada.

## Experimento: até cinco empates por rodada

Todas as partidas de uma rodada usam somente jogos anteriores à data do primeiro jogo da rodada. A diferença máxima entre taxas de gols foi escolhida em 2022 pelo maior F1 macro; o prior de cinco jogos já havia sido escolhido em 2022 no experimento Poisson. Entre os jogos elegíveis, no máximo cinco com menor diferença recebem a classe D; os demais recebem H ou A conforme a maior taxa de gols. A avaliação usa 2023, sem 2024.

As taxas de gols vêm de placares históricos; elas não são xG de finalizações.
A diferença escolhida em 2022 foi de 0.20 gol.

| Método em 2023 | Acurácia | F1 macro | Empates corretos / previstos | H corretos / reais | A corretos / reais |
| --- | ---: | ---: | ---: | ---: | ---: |
| Poisson por rodada | 46,8% | 0,311 | 0/0 | 150/178 | 28/104 |
| Proximidade + teto | 45,5% | 0,381 | 30/101 | 126/178 | 17/104 |

**Conclusão:** A igualdade exata das taxas não ocorreu em 2023. Com diferença máxima de 0,20 gol e teto cinco, a regra encontrou 30 empates, mas criou 71 falsos empates e reduziu acertos em H e A. A rodada 10 teve seis empates reais; o teto de cinco impede reconhecer todos nessa rodada. A regra não substituiu o modelo principal. O resultado é exploratório, pois 2023 já havia sido inspecionado e ainda falta uma temporada nova.

## O que existe hoje

- CSV de 2020 a 2024 preparado e validado, com 1.900 jogos.
- Base com 12 features históricas por partida.
- Primeiro modelo treinado e comparado com uma referência simples.
- Diagnóstico de erros de 2023 e previsões retrospectivas de 2024 gerados localmente.
- Experimento de empates reproduzível, sem trocar o modelo principal.

## Limites conhecidos

- O ganho de acurácia em 2024 foi pequeno, de 47,4% para 48,2%.
- O modelo previu vitória do mandante em 323 dos 380 jogos de 2024.
- Em 2023, reconheceu somente 5 dos 98 empates reais.
- O melhor resultado exploratório para empates ainda precisa de validação em dados novos.
- O Poisson independente não previu nenhum empate pela regra de maior probabilidade em 2023.
- A regra de até cinco empates por rodada aumentou a revocação de D, mas reduziu acertos em H e A.
- Ainda não há fluxo para montar features de uma partida futura.
- A fonte processada cobre apenas até a temporada de 2024.

## Próximas etapas

1. Investigar regras de decisão e correção de placares baixos com novas validações temporais.
2. Reunir uma temporada nova para confirmar se a melhora em empates se repete.
3. Atualizar a fonte de dados e gerar features para partidas não disputadas.
4. Criar uma interface somente após validar o fluxo de previsão futura.

## Como manter este histórico

1. Ao concluir uma mudança, adicione um marco datado em `docs/evolucao.yaml`
   com o arquivo ou commit que comprova o que foi feito.
2. Atualize números e conclusões apenas depois de gerar e conferir as
   saídas do projeto. Registre limites e trabalho pendente.
3. Na raiz do repositório, execute `python docs/build_report.py` e revise
   o diff de `docs/EVOLUCAO.md` e o PDF antes de publicar.
4. Versione juntos o YAML, o Markdown e o PDF.

## Fontes internas desta edição

- Histórico Git do próprio repositório: commits fee7978 a 3066650.
- README.md, src/data/prepare_matches.py e src/model/train_baseline.py.
- artifacts/logistic_baseline_metrics.json e data/processed/predictions_2024.csv, gerados localmente.
- artifacts/validation_2023_diagnostics.json e data/processed/predictions_2023_validation.csv, gerados localmente.
- src/model/experiment_draws.py e artifacts/draw_experiment.json, gerado localmente.
- src/model/experiment_poisson.py e artifacts/poisson_experiment.json, gerado localmente.
- src/model/experiment_round_draws.py e artifacts/round_draw_experiment.json, gerado localmente.
