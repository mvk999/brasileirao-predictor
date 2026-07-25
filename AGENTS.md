# AGENTS.md

## 1. Finalidade deste documento

Este documento define as regras obrigatórias para qualquer pessoa ou agente de inteligência artificial que trabalhe neste repositório.

O objetivo é garantir que o projeto evolua com:

* código simples e legível;
* alterações pequenas e revisáveis;
* testes automatizados;
* segurança;
* rastreabilidade dos dados;
* prevenção de vazamento temporal;
* modelos reproduzíveis;
* documentação atualizada;
* integração contínua;
* responsabilidade humana sobre decisões importantes.

Estas regras prevalecem sobre sugestões automáticas feitas por agentes de IA.

Quando uma solicitação entrar em conflito com este documento, o agente deve interromper a implementação, explicar o conflito e propor uma alternativa compatível.

---

# 2. Visão geral do projeto

## 2.1. Nome provisório

`brasileirao-predictor`

## 2.2. Objetivo

Construir uma aplicação web educacional que:

1. importe partidas e resultados do Campeonato Brasileiro;
2. armazene dados históricos de forma estruturada;
3. calcule características estatísticas dos clubes;
4. treine modelos de machine learning;
5. gere previsões probabilísticas para partidas futuras;
6. registre as previsões antes da realização dos jogos;
7. avalie o desempenho do modelo após os resultados;
8. utilize um LLM apenas para explicar previsões já calculadas;
9. exiba publicamente o histórico e as métricas do sistema.

O projeto deve ser tratado como um sistema real, mesmo tendo finalidade principal de aprendizado.

## 2.3. O que o projeto não é

Este projeto não deve ser utilizado como:

* plataforma de apostas;
* sistema de recomendação financeira;
* ferramenta para prometer resultados;
* fonte de informações privilegiadas;
* mecanismo para garantir vencedores;
* substituto de análise profissional;
* aplicação que apresenta previsões como fatos.

Toda previsão deve ser apresentada como uma estimativa probabilística sujeita a erro.

---

# 3. Princípios fundamentais

Todas as alterações devem seguir estes princípios, nesta ordem:

1. **Correção:** o sistema deve produzir resultados tecnicamente consistentes.
2. **Segurança:** nenhuma entrega deve expor dados, credenciais ou infraestrutura.
3. **Reprodutibilidade:** resultados de treinamento e avaliação devem poder ser reproduzidos.
4. **Simplicidade:** deve ser escolhida a solução mais simples que atenda corretamente ao requisito.
5. **Testabilidade:** comportamentos relevantes devem ser verificáveis automaticamente.
6. **Legibilidade:** o código deve ser compreensível sem depender do autor original.
7. **Rastreabilidade:** deve ser possível identificar a origem de dados, previsões e modelos.
8. **Evolução incremental:** mudanças devem ser pequenas, isoladas e reversíveis.
9. **Observabilidade:** falhas importantes devem produzir logs úteis.
10. **Desempenho:** otimizações devem ser realizadas apenas quando houver necessidade demonstrável.

---

# 4. Stack tecnológica inicial

A arquitetura inicial será um monólito modular.

## 4.1. Tecnologias principais

* Python;
* Django;
* PostgreSQL;
* pandas;
* scikit-learn;
* pytest;
* pytest-django;
* Ruff;
* mypy;
* Docker;
* GitHub Actions;
* provedor de LLM acessado por uma interface desacoplada.

## 4.2. Diretrizes arquiteturais

Não criar microsserviços sem necessidade comprovada.

Não adicionar filas, mensageria, cache distribuído ou serviços externos apenas por possibilidade de uso futuro.

A arquitetura deve começar simples e crescer somente após o surgimento de uma necessidade real.

Aplicar:

* **KISS:** manter as soluções simples;
* **YAGNI:** não implementar funcionalidades especulativas;
* **DRY:** remover duplicações relevantes, sem criar abstrações prematuras;
* **SOLID:** utilizar como orientação, não como justificativa para excesso de classes e interfaces;
* separação clara de responsabilidades;
* dependências apontando para abstrações somente quando isso trouxer benefício concreto.

---

# 5. Estrutura inicial do repositório

A estrutura pode evoluir, mas deve preservar a separação entre dados, previsões, explicações e apresentação.

```text
brasileirao-predictor/
├── AGENTS.md
├── README.md
├── manage.py
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
├── .github/
│   ├── pull_request_template.md
│   └── workflows/
│       └── ci.yml
├── config/
├── football/
│   ├── models.py
│   ├── admin.py
│   ├── selectors/
│   ├── services/
│   ├── management/
│   │   └── commands/
│   └── tests/
├── prediction/
│   ├── models/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── registry/
│   └── tests/
├── explanations/
│   ├── clients/
│   ├── prompts/
│   ├── validators/
│   └── tests/
├── web/
│   ├── views/
│   ├── urls.py
│   └── templates/
└── docs/
    ├── architecture.md
    ├── data-model.md
    ├── model-development.md
    └── decisions/
```

Não criar diretórios genéricos como:

```text
utils/
helpers/
common/
misc/
```

sem uma responsabilidade claramente definida.

Quando um código não possui um local evidente, deve-se primeiro revisar sua responsabilidade. Não se deve esconder falta de organização dentro de um módulo genérico.

---

# 6. Regras de branches

## 6.1. Branch `main`

A branch `main` representa o estado estável, revisado e potencialmente publicável do projeto.

A `main` deve conter apenas código que:

* tenha sido revisado;
* possua testes adequados;
* tenha passado no pipeline de CI;
* esteja documentado quando necessário;
* não contenha erros conhecidos críticos;
* esteja de acordo com este documento.

## 6.2. Proibições na `main`

É proibido:

* realizar commits diretamente na `main`;
* executar `push --force` na `main`;
* apagar a `main`;
* realizar merge com CI falhando;
* realizar merge de código não revisado;
* ignorar testes para liberar uma alteração;
* reduzir verificações de qualidade apenas para fazer o pipeline passar;
* utilizar `--no-verify` para ignorar verificações locais;
* fazer merge de Pull Request em estado de rascunho.

## 6.3. Proteção obrigatória

A `main` deve ser protegida no GitHub com, no mínimo:

* Pull Request obrigatório;
* pelo menos uma aprovação;
* verificações de CI obrigatórias;
* branch atualizada antes do merge;
* conversas de revisão resolvidas;
* bloqueio de force push;
* bloqueio de exclusão;
* histórico linear, quando possível.

## 6.4. Criação de branches

Toda nova alteração deve começar a partir da versão mais recente da `main`.

Exemplo:

```bash
git switch main
git pull --ff-only
git switch -c feature/import-matches
```

## 6.5. Padrão de nomes

Novas funcionalidades:

```text
feature/<descricao-curta>
```

Exemplos:

```text
feature/import-matches
feature/show-next-round
feature/train-logistic-model
feature/generate-predictions
feature/llm-explanations
```

Correções:

```text
fix/<descricao-curta>
```

Exemplos:

```text
fix/duplicate-matches
fix/incorrect-home-score
fix/temporal-data-leakage
```

Refatorações:

```text
refactor/<descricao-curta>
```

Exemplos:

```text
refactor/extract-api-client
refactor/simplify-feature-builder
```

Documentação:

```text
docs/<descricao-curta>
```

Infraestrutura ou manutenção:

```text
chore/<descricao-curta>
ci/<descricao-curta>
```

Correções urgentes em produção:

```text
hotfix/<descricao-curta>
```

## 6.6. Regras de nomenclatura

Nomes de branches devem:

* utilizar letras minúsculas;
* utilizar inglês;
* separar palavras com hífen;
* descrever uma única alteração;
* ser curtos e objetivos.

Não utilizar:

* espaços;
* acentos;
* nomes pessoais;
* números de tarefa sem descrição;
* nomes genéricos como `teste`, `ajustes`, `alteracoes` ou `nova-feature`.

---

# 7. Fluxo obrigatório de desenvolvimento

Para cada tarefa, o agente deve seguir esta sequência:

1. Ler este `AGENTS.md`.
2. Ler os arquivos diretamente relacionados à tarefa.
3. Verificar a arquitetura e os padrões já existentes.
4. Confirmar que a alteração pertence ao escopo do projeto.
5. Atualizar a branch `main`.
6. Criar uma branch específica.
7. Implementar a menor solução completa possível.
8. Criar ou atualizar testes.
9. Executar formatação, lint, tipagem e testes.
10. Revisar o próprio diff.
11. Atualizar a documentação relacionada.
12. Criar commits pequenos e coerentes.
13. Abrir um Pull Request.
14. Aguardar revisão.
15. Corrigir os apontamentos da revisão.
16. Realizar merge somente após aprovação e CI verde.

O agente não deve começar uma implementação extensa sem compreender os arquivos afetados.

---

# 8. Commits

## 8.1. Commits atômicos

Cada commit deve representar uma alteração coerente e funcional.

Um commit não deve misturar:

* funcionalidade nova;
* refatoração não relacionada;
* correção de bug diferente;
* formatação ampla;
* atualização aleatória de dependências.

## 8.2. Estado dos commits

Todo commit deve, sempre que possível:

* compilar;
* executar;
* passar nos testes;
* manter migrations consistentes;
* não quebrar funcionalidades existentes;
* poder ser revertido isoladamente.

Não criar commits propositalmente quebrados com a intenção de corrigir depois.

## 8.3. Mensagens de commit

Utilizar Conventional Commits.

Formatos aceitos:

```text
feat: import historical matches
fix: prevent duplicate match creation
refactor: extract prediction feature builder
test: cover temporal cutoff validation
docs: document model promotion process
chore: update development dependencies
ci: add static type checking
```

A mensagem deve explicar a alteração realizada, não o arquivo modificado.

Evitar:

```text
update files
changes
fix
final
working now
test
```

---

# 9. Pull Requests

Todo Pull Request deve possuir:

* título objetivo;
* contexto do problema;
* descrição da solução;
* arquivos ou módulos principais afetados;
* testes executados;
* possíveis riscos;
* evidências visuais quando houver alteração de interface;
* migrations incluídas;
* documentação atualizada;
* limitações conhecidas.

## 9.1. Tamanho

Pull Requests devem ser pequenos o suficiente para uma revisão cuidadosa.

Quando uma funcionalidade for grande, ela deve ser dividida em etapas independentes, por exemplo:

1. estrutura de dados;
2. cliente da API;
3. importação;
4. cálculo de features;
5. treinamento;
6. geração de previsões;
7. interface.

## 9.2. Revisão obrigatória

Quem implementa não deve considerar sua própria análise suficiente.

O código só pode entrar na `main` após:

* revisão humana;
* aprovação explícita;
* resolução dos comentários;
* aprovação do CI.

Um agente de IA pode preparar um Pull Request, mas não pode aprovar nem realizar sozinho o merge de sua própria implementação.

---

# 10. Padrões de código

## 10.1. Idioma

Utilizar:

* inglês em nomes de classes, funções, variáveis, módulos e branches;
* português nos textos da interface voltados ao usuário;
* português ou inglês na documentação, mantendo consistência dentro de cada documento;
* inglês em mensagens técnicas de log estruturado, quando possível.

## 10.2. Formatação

O código Python deve seguir:

* PEP 8;
* formatação automática;
* imports organizados;
* tipagem estática nas interfaces relevantes;
* docstrings quando adicionarem informação que não está evidente no código.

A formatação automática não deve ser discutida manualmente. O formatador configurado no projeto é a fonte de verdade.

## 10.3. Nomes

Nomes devem descrever intenção.

Preferir:

```python
calculate_recent_form()
training_cutoff
home_win_probability
import_completed_matches()
```

Evitar:

```python
process()
data
value
result2
handle()
do_stuff()
x
temp
```

Nomes curtos só são aceitáveis em escopos muito pequenos e convencionais.

## 10.4. Funções

Funções devem:

* possuir uma responsabilidade principal;
* receber dependências explicitamente quando relevante;
* evitar efeitos colaterais ocultos;
* retornar valores previsíveis;
* possuir nomes compatíveis com o comportamento;
* evitar muitos níveis de indentação;
* falhar de forma explícita quando o erro não puder ser tratado localmente.

Não existe um limite absoluto de linhas, mas funções longas devem ser revisadas criticamente.

Uma função não deve ser dividida apenas para satisfazer uma contagem arbitrária de linhas.

## 10.5. Classes

Uma classe deve representar:

* uma entidade;
* um serviço;
* uma política;
* uma estratégia;
* uma integração;
* outra responsabilidade claramente identificável.

Não criar classes vazias ou interfaces com apenas uma implementação sem necessidade concreta.

## 10.6. Comentários

Comentários devem explicar:

* por que uma decisão não óbvia foi tomada;
* restrições externas;
* riscos;
* regras de negócio;
* comportamentos contraintuitivos.

Comentários não devem repetir o código.

Evitar:

```python
# Incrementa o contador
counter += 1
```

Comentários desatualizados são defeitos e devem ser corrigidos ou removidos.

## 10.7. Complexidade

Antes de adicionar uma abstração, verificar:

* existe duplicação real?
* existem comportamentos diferentes que precisam ser intercambiáveis?
* a abstração reduz ou aumenta o esforço de compreensão?
* a necessidade existe hoje?
* a interface representa um conceito do domínio?

Não criar padrões de projeto apenas para demonstrar conhecimento técnico.

---

# 11. Django

## 11.1. Views

Views devem permanecer pequenas.

Elas podem:

* validar entrada HTTP;
* chamar casos de uso ou serviços;
* selecionar o template ou resposta;
* converter exceções esperadas em respostas adequadas.

Elas não devem concentrar:

* treinamento de modelo;
* importações;
* cálculos estatísticos;
* regras complexas;
* chamadas extensas a APIs externas.

## 11.2. Models

Models devem representar corretamente os dados persistidos e suas invariantes.

Não transformar todos os models em simples estruturas sem comportamento, mas também não concentrar processos inteiros dentro deles.

## 11.3. Services

Services devem representar operações relevantes, como:

```text
ImportMatches
BuildMatchFeatures
TrainPredictionModel
GenerateMatchPrediction
EvaluatePrediction
PromoteModelVersion
GeneratePredictionExplanation
```

Evitar criar um service para operações triviais que já são claras no model ou ORM.

## 11.4. Selectors

Consultas reutilizáveis e de leitura podem ser organizadas em selectors.

Selectors não devem alterar o banco.

## 11.5. Transações

Processos que alteram múltiplos registros relacionados devem avaliar o uso de transações.

Uma falha não pode deixar o banco em um estado parcialmente atualizado sem que isso seja intencional e recuperável.

---

# 12. Banco de dados e migrations

Toda alteração de schema deve possuir migration.

É proibido:

* alterar o banco manualmente sem migration;
* editar uma migration já compartilhada sem justificativa;
* apagar dados silenciosamente;
* remover colunas em uso sem estratégia de migração;
* misturar mudança destrutiva e migração de dados sem análise.

Mudanças destrutivas devem preferir uma estratégia gradual:

1. adicionar a nova estrutura;
2. migrar os dados;
3. adaptar o código;
4. confirmar que a estrutura antiga não é mais utilizada;
5. remover a estrutura antiga em uma alteração posterior.

Constraints importantes devem existir também no banco, não apenas no código Python.

Exemplos:

* identificador externo único;
* partida sem duplicidade;
* probabilidades dentro dos limites esperados;
* relacionamento obrigatório quando aplicável.

---

# 13. Integrações externas

Toda integração deve ser isolada atrás de uma interface ou client específico.

Exemplo:

```python
class FootballDataClient:
    def list_matches(self, season: int) -> list[ExternalMatch]:
        ...
```

A aplicação não deve espalhar chamadas HTTP diretamente por views, models ou comandos.

## 13.1. Requisitos mínimos

Clients externos devem possuir:

* timeout;
* tratamento explícito de erros;
* logs úteis;
* validação da resposta;
* limite de tentativas;
* retry somente para falhas recuperáveis;
* proteção contra criação duplicada;
* identificação da origem dos dados.

## 13.2. Falhas externas

Se a API estiver indisponível:

* dados já armazenados devem continuar acessíveis;
* o sistema não deve apagar dados válidos;
* a falha deve ser registrada;
* a operação deve poder ser repetida com segurança;
* o usuário não deve receber uma mensagem enganosa de sucesso.

## 13.3. Dados externos não são confiáveis

Toda resposta de API deve ser tratada como entrada não confiável.

Validar:

* tipos;
* campos obrigatórios;
* datas;
* identificadores;
* status da partida;
* resultados;
* duplicidades;
* valores inesperados.

---

# 14. Regras de dados

## 14.1. Origem

Todo dado utilizado pelo sistema deve possuir origem identificável.

Quando aplicável, armazenar:

* provedor;
* identificador externo;
* data de coleta;
* temporada;
* data de atualização;
* versão do importador.

## 14.2. Idempotência

Executar uma importação duas vezes não deve cadastrar a mesma partida duas vezes.

Jobs e comandos de importação devem ser idempotentes sempre que possível.

## 14.3. Dados históricos

Não alterar silenciosamente uma partida já finalizada.

Quando o provedor corrigir um resultado histórico:

* registrar a atualização;
* invalidar avaliações afetadas;
* recalcular métricas quando necessário;
* manter rastreabilidade da mudança.

---

# 15. Machine learning

## 15.1. Responsabilidade

O modelo estatístico ou de machine learning é o único componente autorizado a produzir:

* probabilidades de vitória do mandante;
* probabilidade de empate;
* probabilidade de vitória do visitante;
* gols esperados;
* distribuição de placares.

O LLM não pode substituir esse processo.

## 15.2. Prevenção de vazamento temporal

Nenhuma feature pode utilizar informação indisponível antes do horário da partida prevista.

É proibido utilizar:

* resultado da própria partida;
* classificação atualizada após a partida;
* estatísticas calculadas com jogos futuros;
* posição final da temporada;
* dados publicados posteriormente;
* preprocessing ajustado com dados de teste;
* features retroativamente corrigidas sem versionamento.

Toda previsão deve possuir:

```text
generated_at
training_cutoff
feature_cutoff
model_version
data_version
```

Deve ser possível comprovar que:

```text
training_cutoff < match_start_time
feature_cutoff < match_start_time
generated_at < match_start_time
```

## 15.3. Divisão dos dados

Não utilizar divisão aleatória para simular previsão de partidas futuras.

Treino, validação e teste devem respeitar a ordem temporal.

Exemplo:

```text
Treino: temporadas anteriores
Validação: temporada posterior
Teste: período posterior à validação
```

Ou utilizar validação progressiva em janelas temporais.

## 15.4. Preprocessing

Qualquer transformação aprendida com os dados deve ser ajustada somente com o conjunto de treino.

Isso inclui:

* normalização;
* padronização;
* imputação;
* seleção de features;
* encoding;
* redução de dimensionalidade.

Sempre que possível, utilizar pipelines do scikit-learn para manter preprocessing e modelo unidos.

## 15.5. Baseline

Nenhum modelo deve ser considerado útil sem comparação com um baseline.

Exemplos de baseline:

* classe mais frequente;
* vantagem fixa para o mandante;
* média histórica;
* regressão logística simples.

Modelos mais complexos devem demonstrar ganho mensurável.

## 15.6. Métricas

Acurácia não deve ser utilizada isoladamente.

Avaliar, conforme aplicável:

* accuracy;
* log loss;
* Brier score;
* matriz de confusão;
* calibração;
* precision e recall por classe;
* acerto de placar exato como métrica secundária.

## 15.7. Reprodutibilidade

Todo treinamento deve registrar:

* versão do código;
* conjunto de dados;
* período de treinamento;
* features utilizadas;
* hiperparâmetros;
* seed;
* métricas;
* artefato gerado;
* data do treinamento;
* bibliotecas e versões relevantes.

Operações aleatórias devem utilizar seed explícita quando suportado.

## 15.8. Versionamento

Modelos devem ser imutáveis após registrados.

Uma nova execução deve criar uma nova versão, e não sobrescrever silenciosamente a anterior.

Exemplo:

```text
logistic-v1
logistic-v2
poisson-v1
```

## 15.9. Promoção de modelos

Um modelo novo não deve substituir automaticamente o modelo atual apenas por ter sido treinado mais recentemente.

A promoção deve exigir:

* treinamento concluído;
* métricas válidas;
* ausência de vazamento temporal;
* comparação com baseline;
* comparação com o modelo atualmente publicado;
* artefato carregável;
* testes aprovados;
* revisão humana.

## 15.10. Previsões históricas

Uma previsão publicada deve permanecer imutável.

Após o resultado da partida, o sistema pode:

* avaliá-la;
* calcular erro;
* associar métricas;
* exibir o resultado real.

O sistema não pode substituir a previsão antiga por uma nova previsão treinada com informações posteriores.

---

# 16. Uso de LLM

## 16.1. Função permitida

O LLM pode:

* explicar probabilidades;
* resumir fatores fornecidos pelo modelo;
* transformar dados estruturados em texto;
* gerar uma descrição acessível;
* comparar estatísticas previamente calculadas;
* produzir uma explicação em linguagem simples.

## 16.2. Funções proibidas

O LLM não pode:

* definir probabilidades;
* alterar probabilidades;
* inventar estatísticas;
* inventar escalações;
* inventar desfalques;
* inventar notícias;
* afirmar certeza sobre um resultado;
* produzir dados que não estejam na entrada estruturada;
* substituir o modelo de machine learning;
* usar conhecimento próprio como fonte factual sobre a partida;
* recomendar apostas;
* prometer ganhos.

## 16.3. Entrada estruturada

O LLM deve receber um payload explícito, por exemplo:

```json
{
  "match": {
    "home_team": "Palmeiras",
    "away_team": "Fluminense"
  },
  "probabilities": {
    "home_win": 0.52,
    "draw": 0.27,
    "away_win": 0.21
  },
  "expected_goals": {
    "home": 1.61,
    "away": 0.86
  },
  "factors": [
    {
      "name": "home_advantage",
      "description": "Vantagem histórica do mando de campo"
    }
  ]
}
```

## 16.4. Saída estruturada

Sempre que possível, exigir saída estruturada.

Exemplo:

```json
{
  "summary": "O Palmeiras aparece como favorito...",
  "confidence_notice": "A previsão possui incerteza...",
  "mentioned_probabilities": {
    "home_win": 0.52,
    "draw": 0.27,
    "away_win": 0.21
  }
}
```

## 16.5. Validação

Antes de publicar a resposta do LLM:

* validar o schema;
* validar números mencionados;
* impedir probabilidades diferentes das originais;
* rejeitar informações não presentes na entrada;
* aplicar fallback determinístico em caso de falha.

A indisponibilidade do LLM não pode impedir a exibição da previsão estatística.

---

# 17. Testes

Nenhuma funcionalidade é considerada concluída sem testes proporcionais ao risco.

## 17.1. Testes unitários

Devem cobrir:

* regras de negócio;
* cálculos de features;
* métricas;
* validações;
* conversões;
* casos extremos;
* promoção de modelos;
* restrições temporais;
* validação da resposta do LLM.

## 17.2. Testes de integração

Devem cobrir:

* banco de dados;
* importação de partidas;
* migrations;
* comandos Django;
* endpoints;
* templates críticos;
* carregamento de artefatos;
* fluxo de geração e avaliação de previsões.

## 17.3. APIs externas

Testes automatizados comuns não devem depender da disponibilidade de APIs reais.

Utilizar:

* fakes;
* fixtures;
* respostas gravadas e revisadas;
* mocks apenas nas fronteiras externas.

Não mockar a própria regra que está sendo testada.

## 17.4. Testes de machine learning

Devem verificar pelo menos:

* ausência de dados futuros no treino;
* features calculadas somente com dados anteriores;
* split temporal;
* preprocessing ajustado somente no treino;
* soma das probabilidades próxima de `1.0`;
* probabilidades no intervalo `[0, 1]`;
* reprodução com seed;
* carregamento do artefato;
* baseline executável;
* avaliação com dados posteriores ao treino.

## 17.5. Testes do LLM

Não comparar o texto completo palavra por palavra.

Testar:

* estrutura;
* campos obrigatórios;
* números permitidos;
* ausência de afirmações proibidas;
* fallback;
* tratamento de timeout;
* tratamento de resposta inválida.

## 17.6. Correção de bugs

Todo bug corrigido deve, sempre que possível, receber primeiro um teste que reproduza o problema.

O teste deve falhar antes da correção e passar depois dela.

---

# 18. Qualidade automatizada

Antes de abrir um Pull Request, executar:

```bash
ruff format --check .
ruff check .
mypy .
pytest
```

Quando houver migrations:

```bash
python manage.py makemigrations --check --dry-run
python manage.py migrate --check
```

Os comandos definitivos devem permanecer documentados no `README.md` e no CI.

O agente não deve alterar testes corretos apenas porque a implementação atual não consegue passar.

Quando um teste parecer incorreto:

1. explicar por que está incorreto;
2. identificar o comportamento esperado;
3. corrigir o teste e a implementação de forma explícita;
4. registrar a decisão no Pull Request.

---

# 19. CI

O pipeline deve executar em todos os Pull Requests e pushes permitidos.

A ordem recomendada é:

1. instalação reproduzível;
2. verificação de formatação;
3. lint;
4. análise estática;
5. verificação de migrations;
6. testes;
7. relatório de cobertura;
8. verificações de segurança;
9. build da aplicação.

A `main` não pode receber alterações enquanto verificações obrigatórias estiverem falhando.

Não adicionar `continue-on-error` em verificações críticas apenas para deixar o pipeline verde.

---

# 20. Segurança

## 20.1. Segredos

Nunca adicionar ao repositório:

* tokens;
* chaves de API;
* senhas;
* arquivos `.env`;
* credenciais de banco;
* cookies;
* chaves privadas;
* URLs contendo credenciais.

Utilizar variáveis de ambiente.

O arquivo `.env.example` deve conter apenas nomes e exemplos não sensíveis.

## 20.2. Logs

Logs não podem conter:

* tokens;
* senhas;
* headers de autenticação;
* respostas integrais com dados sensíveis;
* variáveis de ambiente;
* prompts contendo segredos.

## 20.3. Dependências

Antes de adicionar uma dependência, avaliar:

* necessidade;
* manutenção;
* licença;
* segurança;
* tamanho;
* compatibilidade;
* possibilidade de uso da biblioteca padrão ou stack atual.

Não adicionar uma biblioteca para resolver um problema trivial.

## 20.4. Entrada de usuários

Toda entrada externa deve ser validada.

Não confiar em:

* query parameters;
* formulários;
* headers;
* arquivos;
* respostas de APIs;
* conteúdo gerado por LLM.

---

# 21. Tratamento de erros

Erros devem ser:

* específicos;
* registrados com contexto;
* apresentados ao usuário sem detalhes internos;
* recuperáveis quando possível.

Não utilizar:

```python
except Exception:
    pass
```

Também não capturar exceções genéricas apenas para esconder falhas.

Quando uma exceção genérica for realmente necessária, ela deve:

* registrar a falha;
* preservar a causa;
* estar em uma fronteira adequada;
* possuir justificativa.

---

# 22. Logs e observabilidade

Operações importantes devem produzir logs estruturados.

Exemplos:

* início e fim de importação;
* quantidade de partidas processadas;
* partidas criadas, atualizadas e ignoradas;
* versão do modelo treinado;
* período utilizado;
* métricas;
* promoção ou rejeição;
* geração de previsões;
* falhas de integração;
* falhas de validação do LLM.

Logs devem permitir investigar falhas sem precisar reproduzi-las imediatamente.

---

# 23. Documentação

Atualizar a documentação quando houver mudança em:

* arquitetura;
* configuração;
* comandos;
* variáveis de ambiente;
* modelo de dados;
* processo de treinamento;
* processo de promoção;
* integração externa;
* regras de negócio;
* limitações conhecidas.

Decisões arquiteturais relevantes devem ser registradas em:

```text
docs/decisions/
```

Exemplo:

```text
docs/decisions/0001-use-django-monolith.md
docs/decisions/0002-llm-only-explains-predictions.md
docs/decisions/0003-use-temporal-validation.md
```

A documentação deve explicar o motivo da decisão, não apenas o resultado.

---

# 24. O que o agente pode fazer

O agente pode:

* analisar o código;
* propor soluções;
* criar branches;
* implementar funcionalidades;
* corrigir bugs;
* criar testes;
* executar testes;
* refatorar código;
* criar migrations;
* atualizar documentação;
* criar commits;
* abrir Pull Requests;
* responder a comentários de revisão;
* sugerir simplificações;
* apontar riscos;
* comparar alternativas técnicas.

---

# 25. O que o agente não pode fazer

O agente não pode:

* realizar push diretamente na `main`;
* fazer merge sem revisão humana;
* desativar proteção de branch;
* ignorar CI;
* remover testes para esconder regressões;
* alterar requisitos silenciosamente;
* criar funcionalidades fora do escopo;
* introduzir dependências desnecessárias;
* expor segredos;
* alterar dados de produção sem autorização;
* executar migrations destrutivas sem revisão;
* sobrescrever modelos registrados;
* recalcular previsões históricas usando dados futuros;
* utilizar resultados futuros no treinamento;
* permitir que o LLM invente dados;
* tratar conteúdo gerado por LLM como fonte confiável;
* recomendar apostas;
* afirmar que uma previsão é garantida;
* criar arquitetura excessivamente complexa sem necessidade demonstrada;
* realizar grandes refatorações não relacionadas à tarefa atual;
* ocultar limitações ou incertezas.

---

# 26. Comportamento esperado do agente

O agente deve atuar como colaborador técnico, e não como executor sem julgamento.

Ele deve:

* questionar requisitos contraditórios;
* sinalizar risco de segurança;
* sinalizar vazamento de dados;
* sinalizar complexidade desnecessária;
* preferir alterações pequenas;
* preservar comportamentos existentes;
* verificar o impacto da mudança;
* explicar decisões não evidentes;
* admitir quando não possui informação suficiente;
* não inventar detalhes técnicos;
* não assumir que o código gerado está correto;
* revisar a própria implementação antes de apresentá-la.

O agente não deve concordar automaticamente com toda solicitação.

Quando existir uma solução mais simples, segura ou correta, deve apresentá-la de forma objetiva.

---

# 27. Refatoração

Refatoração deve ser contínua e incremental.

É permitido refatorar durante uma feature quando isso for necessário para implementar a mudança com segurança.

Entretanto, a refatoração deve permanecer relacionada ao escopo do Pull Request.

Sinais que justificam revisão:

* duplicação relevante;
* função com múltiplas responsabilidades;
* acoplamento excessivo;
* regra de negócio em view;
* chamada externa espalhada;
* nomes imprecisos;
* condicionais profundamente aninhadas;
* dependências circulares;
* arquivo crescendo sem organização;
* testes difíceis devido ao design.

Não realizar uma reescrita completa quando uma alteração incremental resolver o problema.

---

# 28. Definição de pronto

Uma tarefa só está pronta quando:

* [ ] o requisito foi atendido;
* [ ] a solução está na branch correta;
* [ ] não houve commit direto na `main`;
* [ ] o código está formatado;
* [ ] lint está aprovado;
* [ ] tipagem está aprovada;
* [ ] testes estão aprovados;
* [ ] novos comportamentos possuem testes;
* [ ] bugs possuem teste de regressão quando aplicável;
* [ ] migrations foram verificadas;
* [ ] não existem segredos no diff;
* [ ] erros são tratados adequadamente;
* [ ] logs necessários foram adicionados;
* [ ] documentação foi atualizada;
* [ ] o diff foi revisado;
* [ ] o Pull Request está completo;
* [ ] o CI está verde;
* [ ] os comentários foram resolvidos;
* [ ] houve aprovação humana;
* [ ] a alteração pode ser revertida com segurança.

Para funcionalidades de machine learning:

* [ ] não existe vazamento temporal;
* [ ] existe baseline;
* [ ] o split é temporal;
* [ ] seeds relevantes foram registradas;
* [ ] dados e features são rastreáveis;
* [ ] métricas foram registradas;
* [ ] o modelo foi versionado;
* [ ] previsões históricas permanecem imutáveis.

Para funcionalidades com LLM:

* [ ] a entrada é estruturada;
* [ ] a saída é validada;
* [ ] números não podem ser modificados;
* [ ] informações não podem ser inventadas;
* [ ] existe timeout;
* [ ] existe fallback;
* [ ] a aplicação funciona sem o LLM.

---

# 29. Regra final

Velocidade não é medida pela quantidade de código produzido.

Velocidade sustentável significa:

* entregar pequenas partes funcionais;
* detectar erros rapidamente;
* manter a base compreensível;
* preservar confiança nos dados;
* conseguir alterar o sistema sem medo;
* chegar à produção sem acumular dívida técnica desnecessária.

A IA deve acelerar a execução.

A responsabilidade por direção, arquitetura, segurança, qualidade e aprovação permanece humana.

