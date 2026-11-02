import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import database as db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    with db.connect(tmp_path / "test.db") as c:
        yield c


@pytest.fixture(scope="session")
def text_model():
    """Train once per test session on the bundled data (a few seconds)."""
    from src.text_classifier import train_default_model
    return train_default_model()
