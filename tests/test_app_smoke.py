"""Run every Streamlit page headlessly (streamlit.testing) against a throw-away database."""
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import database as db
from src import transactions as tx
from src.synthetic_data import generate_transactions

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["pages/1_Transactions.py", "pages/2_Dashboard.py", "pages/3_Insights.py",
         "pages/4_Smart_Categoriser.py", "pages/5_Settings.py"]


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("SEA_DB_PATH", str(path))
    return path


@pytest.fixture
def demo_db(empty_db):
    with db.connect(empty_db) as c:
        tx.bulk_add(c, generate_transactions(end=date.today()))
    return empty_db


def run(script: str) -> AppTest:
    return AppTest.from_file(str(ROOT / script), default_timeout=60).run()


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


@pytest.mark.parametrize("script", ["app.py"] + PAGES)
def test_pages_render_on_empty_database(empty_db, script):
    at = run(script)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("script", ["app.py"] + PAGES)
def test_pages_render_with_demo_data(demo_db, script):
    at = run(script)
    assert not at.exception, [e.value for e in at.exception]


def test_add_expense_with_ml_suggestion(empty_db):
    at = run("app.py")
    at.text_input(key="f_amount").set_value("250")
    at.text_input(key="f_desc").set_value("Late night food delivery").run()
    assert any("Food" in s.value for s in at.success)                       # suggestion card shown
    button(at, "Use this suggestion").click().run()
    assert at.selectbox(key="f_category").value == "Food"
    assert at.selectbox(key="f_needwant").value == "Want"
    button(at, "Save transaction").click().run()
    assert not at.exception
    with db.connect(empty_db) as c:
        df = tx.load_dataframe(c)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.amount_paise, r.category, r.need_want, r.predicted_category) == (25000, "Food", "Want", "Food")
    assert at.text_input(key="f_amount").value == ""                        # form reset after save


def test_invalid_entry_shows_errors_and_saves_nothing(empty_db):
    at = run("app.py")
    at.text_input(key="f_amount").set_value("-5")
    button(at, "Save transaction").click().run()
    assert any("Amount" in e.value for e in at.error)
    with db.connect(empty_db) as c:
        assert tx.load_dataframe(c).empty


def test_smart_categoriser_example_output(empty_db):
    at = run("pages/4_Smart_Categoriser.py")
    assert any("Food" in s.value and "discretionary" in s.value for s in at.success)
    at.text_input(key="sc_text").set_value("asdfgh").run()
    assert any("Not sure" in w.value for w in at.warning)


def test_settings_demo_data_and_delete_guard(empty_db):
    at = run("pages/5_Settings.py")
    button(at, "Add demo data").click().run()
    with db.connect(empty_db) as c:
        assert len(tx.load_dataframe(c)) > 50
    at = run("pages/5_Settings.py")
    assert button(at, "Delete all my transactions").disabled                # needs the typed confirmation
    at.text_input(key="del_confirm").set_value("DELETE").run()
    button(at, "Delete all my transactions").click().run()
    with db.connect(empty_db) as c:
        assert tx.load_dataframe(c).empty
