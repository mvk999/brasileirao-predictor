from unittest.mock import patch

import pytest
from django.db import OperationalError, connection
from django.test import Client


@pytest.mark.django_db
def test_health_check_returns_ok(client: Client) -> None:
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.django_db
def test_postgresql_connection_executes_query() -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")

        assert cursor.fetchone() == (1,)


def test_health_check_reports_database_failure(client: Client) -> None:
    with patch("config.health.connection.cursor", side_effect=OperationalError):
        response = client.get("/health/")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
