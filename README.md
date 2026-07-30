# Brasileirão Predictor

Fundação Django do projeto `brasileirao-predictor`, executada com PostgreSQL
por meio do Docker Compose.

## Configuração

Crie o arquivo local de variáveis de ambiente:

```bash
cp .env.example .env
```

Os valores de `.env.example` são apenas para desenvolvimento local.

## Execução

Construa as imagens e inicie a aplicação:

```bash
docker compose up --build
```

A aplicação ficará disponível em <http://localhost:8000> e o endpoint de
verificação em <http://localhost:8000/health/>.

Em outro terminal, execute as migrations iniciais:

```bash
docker compose exec web python manage.py migrate
```

## Verificações

Com os containers em execução:

```bash
docker compose exec web ruff format --check .
docker compose exec web ruff check .
docker compose exec web pytest
docker compose exec web python manage.py check
docker compose exec web python manage.py makemigrations --check --dry-run
```

## Encerramento

Interrompa e remova os containers:

```bash
docker compose down
```
