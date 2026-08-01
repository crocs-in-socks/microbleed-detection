"""Domain-error handling for the CLI layer.

The core and pipeline layers raise precise precondition errors. The CLI catches
the *expected* domain errors, presents them cleanly, and exits nonzero. Anything
unexpected is left to surface as a traceback so it is not silently hidden.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

import typer

logger = logging.getLogger(__name__)

# Exceptions the pipelines and core deliberately raise for invalid inputs or
# unmet preconditions. These are user-actionable, so we format them rather than
# dumping a traceback.
EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    FileNotFoundError,
    EnvironmentError,
)


@contextmanager
def domain_errors() -> Iterator[None]:
    """Convert expected domain errors into a clean, nonzero CLI exit.

    Unexpected exception types are not caught here and will surface normally.
    """
    try:
        yield
    except EXPECTED_ERRORS as error:
        logger.error("%s", error)
        raise typer.Exit(code=1) from error
