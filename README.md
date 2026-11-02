# 💸 Smart Expense Analyzer

A personal expense tracker with exact, rule-based reporting **plus** an NLP feature,
**Intelligent Transaction Understanding**, that reads a free-text purpose such as
*"Late night food delivery"* or *"friend ke saath dinner"* and suggests the **category** and the
likely **spending context** (essential / discretionary / mixed) with a confidence score and honest limitations.

Stack: Python · Streamlit · Pandas · Plotly · SQLite · scikit-learn · pytest

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

First thing to try: **Settings → Add demo data** (fake data only), then explore the pages.
The trained model ships in `models/`; if it is missing the app trains one automatically in a few seconds.

```bash
pytest -q                                # 104 tests (validation, hand-calculated totals, ML, every page)
python scripts/train_model.py            # retrain + evaluate the ML model -> reports/ml_report.md
python scripts/evaluate_anomaly.py       # rule baseline vs IsolationForest -> reports/anomaly_report.md
python scripts/generate_sample_data.py   # rewrite data/sample_synthetic_transactions.csv
```

## What it does (maps to the project document)

| Doc section | Implementation |
|---|---|
| 5.1 Daily entry form | `app.py` - date, amount, type, category, description, wasteful?, need/want, payment, context, notes. Only date, amount, category are mandatory. |
| 5.2 History | `pages/1_Transactions.py` - filter (date, category, type, wasteful, payment), search, sort, pagination, edit, delete with confirmation, CSV export |
| 5.3 Dashboard | `pages/2_Dashboard.py` - totals, daily average (shows days used), category share, month-over-month, largest expenses, wasteful-labelled spend, income & net cash-flow |
| 5.4 Wasteful reflection | Only sums what **you** labelled "Yes"; the app never judges a purchase |
| 5.6 Insights | `pages/3_Insights.py` - category increase vs the comparable days last month, unusually large expense, frequent small purchases, wasteful summary, optional logging reminder. Every alert has a "How was this calculated?" rule. "Not enough data yet" is shown instead of fake personalisation |
| 6 ML role | Rule-based anomaly baseline + experimental IsolationForest, evaluated side by side; and the new **text understanding model** below |
| 7 Architecture | `src/` modules: `database`, `validation`, `transactions`, `analytics`, `insights`, `anomaly_detection`, + `text_classifier`, `training_data` |
| 9 Privacy | Local SQLite, parameterised SQL, CSV-injection-safe export, delete-all with typed confirmation, synthetic demo data only |
| 11 Testing | `tests/` - includes the 11.2 scenario (23 synthetic transactions over 2 months, totals verified by hand, edit/delete, leap-year and month-boundary cases, empty DB) |

Business rules (also asserted in tests): amounts are stored as **integer paise** (no float drift); `expense` adds to
spending; `refund` subtracts from its own category; `income` is never counted as spending; periods are inclusive;
daily average = net spending ÷ calendar days in the period; transfers between own accounts are not modelled (don't record them).

## 🧠 Intelligent Transaction Understanding (the ML feature)

**Input:** a short description, English or romanised Hinglish.
**Output:** category + spending context + confidence + the words that drove the decision + limitations.

```
Input : "Late night food delivery"
Output: Food, likely discretionary (want) - category confidence 86% (high)
        decided by: "late night", "food"
        limits: need/want is a statistical guess from words only; trained on synthetic data
```

**How it works** (`src/text_classifier.py`)

1. **NLP pre-processing** - lower-case, strip amounts/numbers/punctuation (so "Rs 5000" is not evidence for anything).
2. **TF-IDF** - word 1–2-grams *and* character 2–5-grams, so typos and Hinglish spellings (`kharidi`, `khareede`) still overlap.
3. **Logistic Regression** - two heads on the same feature recipe: *category* (9 classes) and *context* (essential / discretionary / mixed).
4. **Trust layer** - probabilities → `high / medium / low` confidence; "words understood" ratio; close-call detection;
   gibberish and empty text are rejected as *low* and never pre-filled. In the UI the user always confirms or overrides.
5. **Feedback loop** - the app stores what it suggested; **Settings** shows how often you agreed and can retrain with your own confirmed entries.

**Evaluation** (full detail: `reports/ml_report.md`, confusion matrices included)

| Evaluation | Category acc / macro-F1 | Context acc / macro-F1 |
|---|---|---|
| Grouped 5-fold CV - held-out *phrases* (harsh, new vocabulary) | 0.51 / 0.49 | 0.53 / 0.52 |
| Independent hand-written test set (79 phrases, zero overlap with training) | 0.94 / 0.94 | 0.85 / 0.84 |
| Majority-class baseline (test) | 0.18 / 0.03 | - |

Confidence is meaningful: in cross-validation, predictions with confidence <0.35 were right 19% of the time, 0.35–0.60 → 45%, ≥0.60 → 71%
(and ≥0.90 → 84%). On the test set, high-confidence predictions were right 97% of the time.

**Honest reading of these numbers**

* There is **no public labelled dataset** of Indian personal-expense descriptions, so the training data is ~317 hand-written seed phrases
  (English + Hinglish) augmented to ~3,800 rows. The test set was written by the same author, so the 0.94 is the *optimistic* end and the CV 0.51 the
  *pessimistic* end (every held-out phrase has unfamiliar words). Expect real-world accuracy in between - and verify with the in-app acceptance rate.
* A simple Complement Naive Bayes on the same features scored slightly *higher* on the test set (category acc 0.97 vs 0.94) and the same in CV.
  Logistic Regression was kept because the project asked for it and it gives usable per-class probabilities, but the difference is not meaningful on 79 phrases.
* "Context" labels (need/want) are the author's view of the *typical* nature of a purchase. The same purchase can be a need for one person and a want for another,
  which is why it is always shown as *likely* and the user's own labels stay separate.
* Not handled: Devanagari script, other languages, amounts/time/merchant codes as signals, negation.
* **Next step with real data:** compare against sentence-embedding features (e.g. a multilingual MiniLM) - only adopt them if they beat this baseline on the same held-out data.

## Unusual-expense detection (measured, not assumed)

`python scripts/evaluate_anomaly.py` injects known anomalies (4–8× the normal amount) into 10 synthetic one-year histories:

| detector | precision | recall | F1 |
|---|---|---|---|
| Rule baseline: above max(Q3+3×IQR, 3×median) of *earlier* same-category entries | 0.49 | 0.60 | 0.54 |
| IsolationForest (experimental, in-sample, true contamination given) | 0.74 | 0.53 | 0.61 |

The thresholds were tuned on separate seeds (an initial 1.5×IQR rule had precision ≈ 0.3). IsolationForest scores higher here, but it is unsupervised, fitted in-sample and was
told the true anomaly rate, so the rule stays the default (explainable, works with little data, no look-ahead). Both are exposed in the Insights page. "Unusual" never means fraud.

## Project layout

```
app.py                       home: quick entry + ML suggestion
pages/                       Transactions · Dashboard · Insights · Smart_Categoriser · Settings
src/
  config.py validation.py database.py transactions.py
  analytics.py insights.py anomaly_detection.py synthetic_data.py
  text_classifier.py training_data.py ui.py
scripts/                     train_model.py · evaluate_anomaly.py · generate_sample_data.py
tests/                       pytest suite (incl. headless Streamlit page tests)
data/                        handwritten_test_set.csv · training_descriptions.csv · sample_synthetic_transactions.csv
models/                      text_classifier.joblib · metrics.json
reports/                     ml_report.md · anomaly_report.md · confusion matrices
```

## Scope decisions & limitations

* **Single user, no login** - the document allows deferring authentication. If you add accounts later: hash passwords (e.g. argon2/bcrypt), add `user_id` to
  `transactions`, and filter **every** query by it.
* **No bank/CSV import, OCR, budgets or forecasting** (document: out of scope for v1). Budgets and a forecast-vs-baseline test are the natural next steps.
* Fixed category list (a `categories` table for custom categories is a later change). Groceries are under **Food**.
* **Deployment:** Streamlit Community Cloud's filesystem is ephemeral - use synthetic data for a public demo; for real persistence use a hosted database.
  Never commit `data/*.db` (it is git-ignored).
* Not a bank, financial adviser or fraud-detection system. Don't enter passwords, card numbers, PINs or OTPs.
