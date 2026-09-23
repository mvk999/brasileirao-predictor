# Evolução do Brasileirão Predictor

**Edição 1 · Atualizado em 23/09/2026**

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

## O que existe hoje

- CSV de 2020 a 2024 preparado e validado, com 1.900 jogos.
- Base com 12 features históricas por partida.
- Primeiro modelo treinado e comparado com uma referência simples.
- Métricas e previsões retrospectivas de 2024 geradas localmente.

## Limites conhecidos

- O ganho de acurácia em 2024 foi pequeno, de 47,4% para 48,2%.
- O modelo previu vitória do mandante em 323 dos 380 jogos de 2024.
- Ainda não há fluxo para montar features de uma partida futura.
- A fonte processada cobre apenas até a temporada de 2024.

## Próximas etapas

1. Investigar erros por classe, principalmente empates e vitórias do visitante.
2. Comparar novas features e métodos usando novas validações temporais.
3. Atualizar a fonte de dados e gerar features para partidas não disputadas.
4. Criar uma interface somente após validar o fluxo de previsão futura.

## Como manter este histórico

1. Ao concluir uma mudança, adicione um marco datado em `docs/evolucao.yaml`
   com o commit que comprova o que foi feito.
2. Atualize números e conclusões apenas depois de gerar e conferir as
   saídas do projeto. Registre limites e trabalho pendente.
3. Na raiz do repositório, execute `python docs/build_report.py` e revise
   o diff de `docs/EVOLUCAO.md` e o PDF antes de publicar.
4. Versione juntos o YAML, o Markdown e o PDF.

## Fontes internas desta edição

- Histórico Git do próprio repositório: commits fee7978 a c544fac.
- README.md, src/data/prepare_matches.py e src/model/train_baseline.py.
- artifacts/logistic_baseline_metrics.json e data/processed/predictions_2024.csv, gerados localmente.
