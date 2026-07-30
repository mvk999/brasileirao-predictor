import logging

from django.db import DatabaseError, connection
from django.http import HttpRequest, JsonResponse

logger = logging.getLogger(__name__)


def health_check(_request: HttpRequest) -> JsonResponse:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            database_result = cursor.fetchone()
    except DatabaseError:
        logger.exception("Database health check failed")
        return JsonResponse({"status": "unavailable"}, status=503)

    if database_result != (1,):
        logger.error("Database health check returned an unexpected result")
        return JsonResponse({"status": "unavailable"}, status=503)

    return JsonResponse({"status": "ok"})
