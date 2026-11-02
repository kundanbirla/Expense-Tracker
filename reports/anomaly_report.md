# Anomaly detector comparison

Evaluated on 10 synthetic one-year histories: 5877 expenses, 171 injected anomalies.

| detector | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| rule | 0.49 | 0.60 | 0.54 | 102 | 106 | 69 |
| rule (rows with enough history) | 0.49 | 0.66 | 0.56 | 102 | 106 | 53 |
| isolation_forest | 0.74 | 0.53 | 0.61 | 90 | 32 | 81 |

Caveats: injected anomalies are simple multiples of a normal amount, which favours both detectors; the rule baseline cannot flag the first 8 entries of a category (so its plain recall is capped), and IsolationForest is fitted in-sample with the true contamination rate supplied. Real data will behave worse - review alerts manually.
