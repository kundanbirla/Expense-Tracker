"""Intelligent Transaction Understanding.

Given a free-text purpose ("Late night food delivery", "friend ke saath dinner") predict

  1. the expense **category**                       (9 classes)
  2. the likely **spending context**                (essential / discretionary / mixed)

together with calibrated-looking probabilities, the words that drove the decision, and an honest list
of limitations.

Technique: NLP pre-processing -> TF-IDF (word 1-2 grams + character 2-5 grams, so typos and Hinglish
spellings still overlap) -> multinomial Logistic Regression. Two independent models share the same
feature recipe. The "context" prediction is a *suggestion about the type of purchase*, never a verdict
on the user's behaviour - that is why the UI always lets the user override it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config

MODEL_VERSION = 1

CONTEXT_LABELS = {
    "essential": "Likely essential (need)",
    "discretionary": "Likely discretionary (want)",
    "mixed": "Could be either need or want",
}
CONTEXT_TO_NEED_WANT = {"essential": "Need", "discretionary": "Want", "mixed": "Both"}
NEED_WANT_TO_CONTEXT = {v: k for k, v in CONTEXT_TO_NEED_WANT.items()}

# Confidence bands (tuned on out-of-fold predictions, NOT on the hand-written test set).
HIGH_CONF = 0.60
MEDIUM_CONF = 0.35
CLOSE_CALL_MARGIN = 0.10


# Filler words are never shown as "the words that drove the decision".
_FILLER = frozenset("a an the and of to for in on at with from by is was ke ki ka ko se me mein liye saath "
                    "aur ek ye wo hai tha paid bought spent today yesterday aaj online upi cash this week "
                    "rs inr card".split())


# ------------------------------------------------------------------ NLP
_AMOUNT = re.compile(r"(?:₹|\brs\b\.?|\binr\b)\s*\d[\d,]*(?:\.\d+)?", re.IGNORECASE)


def preprocess(text: str) -> str:
    """Lower-case, drop amounts / numbers / punctuation, collapse whitespace.

    Amounts are removed on purpose: "Rs 250" is not evidence for any category.
    Romanised Hinglish is supported; Devanagari script is not.
    """
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _AMOUNT.sub(" ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def build_pipeline(C: float = 10.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=preprocess, analyzer="word", ngram_range=(1, 2),
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
    return Pipeline([("tfidf", features), ("clf", clf)])


# ------------------------------------------------------------ prediction
@dataclass
class Prediction:
    text: str
    category: str
    category_confidence: float
    category_alternatives: list[tuple[str, float]]       # top-3 including the winner
    context: str                                          # essential | discretionary | mixed
    context_confidence: float
    context_probabilities: dict[str, float]
    confidence_level: str                                 # high | medium | low
    suggest: bool                                         # safe to pre-fill the form?
    top_terms: list[str]
    known_token_ratio: float
    limitations: list[str] = field(default_factory=list)

    @property
    def context_label(self) -> str:
        return CONTEXT_LABELS[self.context]

    @property
    def need_want(self) -> str:
        return CONTEXT_TO_NEED_WANT[self.context]

    def summary(self) -> str:
        return (f"{self.category}, {self.context_label.lower()} — "
                f"confidence {self.category_confidence:.0%} ({self.confidence_level})")


class TransactionTextModel:
    """TF-IDF + Logistic Regression for category and spending context."""

    def __init__(self, C: float = 10.0):
        self.C = C
        self.category_pipe: Pipeline | None = None
        self.context_pipe: Pipeline | None = None
        self.meta: dict = {}
        self._vocab: frozenset[str] | None = None
        self._names: np.ndarray | None = None

    # -- training -------------------------------------------------------
    def fit(self, texts: Sequence[str], categories: Sequence[str],
            context_texts: Sequence[str] | None = None,
            contexts: Sequence[str] | None = None) -> "TransactionTextModel":
        """Fit both heads. Context labels may come from a different (partly labelled) subset."""
        if context_texts is None:
            context_texts, contexts = texts, contexts
        if contexts is None:
            raise ValueError("contexts must be provided")
        self.category_pipe = build_pipeline(self.C).fit(list(texts), list(categories))
        self.context_pipe = build_pipeline(self.C).fit(list(context_texts), list(contexts))
        self._vocab = self._names = None
        self.meta = {
            "model_version": MODEL_VERSION,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_category_rows": len(texts),
            "n_context_rows": len(context_texts),
            "sklearn_version": sklearn.__version__,
        }
        return self

    # -- internals ------------------------------------------------------
    def _check(self) -> None:
        if self.category_pipe is None or self.context_pipe is None:
            raise RuntimeError("Model is not trained.")

    def _vocabulary(self) -> frozenset[str]:
        if self._vocab is None:
            vec = self.category_pipe.named_steps["tfidf"].transformer_list[0][1]
            self._vocab = frozenset(k for k in vec.vocabulary_ if " " not in k)
        return self._vocab

    def _feature_names(self) -> np.ndarray:
        if self._names is None:
            self._names = self.category_pipe.named_steps["tfidf"].get_feature_names_out()
        return self._names

    def _top_terms(self, text: str, class_index: int, k: int = 3) -> list[str]:
        X = self.category_pipe.named_steps["tfidf"].transform([text])
        coef = self.category_pipe.named_steps["clf"].coef_[class_index]
        contrib = X.multiply(coef).tocoo()
        names = self._feature_names()
        scored: dict[str, float] = {}
        for col, val in zip(contrib.col, contrib.data):
            name = names[col]
            term = name[len("word__"):]
            if name.startswith("word__") and val > 0 and not all(w in _FILLER for w in term.split()):
                scored[term] = scored.get(term, 0.0) + float(val)
        return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]

    # -- inference ------------------------------------------------------
    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        self._check()
        texts = [str(t) for t in texts]
        if not texts:
            return []
        cat_proba = self.category_pipe.predict_proba(texts)
        ctx_proba = self.context_pipe.predict_proba(texts)
        cat_classes = self.category_pipe.named_steps["clf"].classes_
        ctx_classes = self.context_pipe.named_steps["clf"].classes_
        vocab = self._vocabulary()
        out = []
        for i, text in enumerate(texts):
            tokens = preprocess(text).split()
            known = [t for t in tokens if t in vocab]
            ratio = len(known) / len(tokens) if tokens else 0.0

            order = np.argsort(-cat_proba[i])
            p1 = float(cat_proba[i][order[0]])
            p2 = float(cat_proba[i][order[1]]) if len(order) > 1 else 0.0
            alts = [(str(cat_classes[j]), float(cat_proba[i][j])) for j in order[:3]]
            ci = int(np.argmax(ctx_proba[i]))
            ctx_p = {str(c): float(p) for c, p in zip(ctx_classes, ctx_proba[i])}

            if ratio == 0.0 or p1 < MEDIUM_CONF:
                level = "low"
            elif p1 >= HIGH_CONF and ratio >= 0.5:
                level = "high"
            else:
                level = "medium"

            pred = Prediction(
                text=text, category=alts[0][0], category_confidence=p1, category_alternatives=alts,
                context=str(ctx_classes[ci]), context_confidence=float(ctx_proba[i][ci]),
                context_probabilities=ctx_p, confidence_level=level, suggest=level != "low",
                top_terms=self._top_terms(text, int(order[0])) if ratio > 0 else [],
                known_token_ratio=ratio,
            )
            pred.limitations = self._limitations(pred, tokens, [t for t in tokens if t not in vocab], p1 - p2)
            out.append(pred)
        return out

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    @staticmethod
    def _limitations(p: Prediction, tokens: list[str], unknown: list[str], margin: float) -> list[str]:
        notes = []
        if not tokens:
            notes.append("The description is empty or has no usable words - nothing to base a prediction on.")
        elif p.known_token_ratio == 0.0:
            notes.append("None of these words were seen in training, so this is effectively a guess.")
        elif unknown:
            notes.append("Words the model has never seen (ignored): " + ", ".join(unknown[:5]) + ".")
        if len(tokens) == 1 and p.known_token_ratio > 0:
            notes.append("Very short description - adding detail (what, where, with whom) improves accuracy.")
        if p.confidence_level == "low":
            notes.append("Low confidence - please choose the category yourself.")
        elif margin < CLOSE_CALL_MARGIN and len(p.category_alternatives) > 1:
            notes.append(f"Close call between {p.category_alternatives[0][0]} and {p.category_alternatives[1][0]}.")
        if p.context_confidence < 0.5:
            notes.append("The need/want guess is uncertain (below 50%).")
        notes.append("Need/want is a statistical guess from the words only - you know your real intent; "
                     "it is never a judgement of the purchase.")
        notes.append("The model was trained mainly on synthetic English/Hinglish examples; accuracy on "
                     "your real descriptions may be lower. Devanagari script is not supported.")
        return notes

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path = config.MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path = config.MODEL_PATH) -> "TransactionTextModel":
        model = joblib.load(path)
        if not isinstance(model, TransactionTextModel):
            raise TypeError("File does not contain a TransactionTextModel.")
        return model


# ------------------------------------------------------- convenience API
def train_default_model(extra=None, n_aug: int = 12):
    """Train on the bundled synthetic data (+ optional user-confirmed rows).

    ``extra``: DataFrame with columns ``text``, ``category`` and optional ``context``
    (essential/discretionary/mixed). Rows without a context only train the category head.
    """
    from .training_data import build_training_frame

    frame = build_training_frame(n_aug=n_aug)
    texts, cats = frame["text"].tolist(), frame["category"].tolist()
    ctx_texts, ctxs = frame["text"].tolist(), frame["context"].tolist()
    if extra is not None and len(extra):
        extra = extra.dropna(subset=["text", "category"])
        extra = extra[extra["text"].astype(str).str.strip() != ""]
        texts += extra["text"].tolist()
        cats += extra["category"].tolist()
        if "context" in extra:
            labelled = extra.dropna(subset=["context"])
            ctx_texts += labelled["text"].tolist()
            ctxs += labelled["context"].tolist()
    return TransactionTextModel().fit(texts, cats, ctx_texts, ctxs)


def load_or_train(path: str | Path = config.MODEL_PATH) -> TransactionTextModel:
    """Load the saved model; (re)train on the bundled data if missing or incompatible."""
    path = Path(path)
    if path.exists():
        try:
            model = TransactionTextModel.load(path)
            # Pickles are not a stable interchange format across scikit-learn
            # releases.  In particular, LogisticRegression objects saved by
            # 1.9.x can load under 1.7.x while missing attributes that
            # ``predict_proba`` expects (such as ``multi_class``).  Treat a
            # version mismatch as an incompatible artifact and rebuild it.
            if (model.meta.get("model_version") == MODEL_VERSION
                    and model.meta.get("sklearn_version") == sklearn.__version__):
                return model
        except Exception:  # corrupted / sklearn-version mismatch -> retrain below
            pass
    model = train_default_model()
    try:
        model.save(path)
    except OSError:  # read-only deployment filesystem: keep the in-memory model
        pass
    return model
