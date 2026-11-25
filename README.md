# Fairness: which error, compared with whom?

Interactive Marimo WASM app for teaching algorithmic fairness with COMPAS: which
error matters (numerator), whose risk you compare (denominator), and what closing
the gap between groups costs in accuracy. It follows slides 10–20 of the *Deep
learning advanced* lecture of ML4Prediction.

## Browser app

**[Open the app](https://javier.science/marimo_fairness/)**
GitHub Actions checks the app, builds the WASM bundle, and deploys it to GitHub
Pages on every push to `main`. It runs entirely in the browser, with no server
or external data.

## Run locally

```bash
uv sync
sh run.sh            # local server
sh export_wasm.sh    # static browser version in build/
python -m http.server --directory build
```

Only depends on marimo (`pyproject.toml`, `uv.lock`; `app.py` also carries script
metadata for `uvx marimo run --sandbox app.py`).

## On the page

1. **Cut-offs and confusion matrices.** One slider per group; everyone with a score
   at or above the cut-off is flagged.
2. **Bars per group:** selection rate (independence), FPR and FNR (separation),
   FDR and FOR (sufficiency), and balanced accuracy, with the gap between groups.
3. **Step 1 · Numerator:** false positives or false negatives, each with what it means
   in a punitive and an assistive use. Bars that don't count that error fade out.
4. **Step 2 · Denominator:** equal chance of being misjudged (condition on the true
   outcome) or equal reliability of the label (condition on the prediction). One
   metric remains; the confusion matrices outline its numerator and fade everyone
   outside its denominator.
5. **Step 3 · Pareto plot:** every pair of cut-offs as a point (gap in the chosen
   metric vs balanced accuracy of both groups), the Pareto front labelled with its
   cut-offs, and the same-cut-off pairs. Hovering over a point shows its cut-offs,
   gap and balanced accuracy.
6. **Why can't you have it all?** Chouldechova's (2017) identity, checked live:
   FPR = p/(1−p) · FDR/(1−FDR) · (1−FNR).

With COMPAS, equalising FPR costs about 1 point of balanced accuracy (4/6) and
equalising FDR costs nothing (6/5). The price is a different cut-off per group.

## COMPAS data

Embedded as 40 aggregate counts (5,278 defendants), no individual records.
Source: [ProPublica COMPAS analysis](https://github.com/propublica/compas-analysis),
`compas-scores-two-years.csv`, blob `d5c382e68132f84f5bf901c1c53ab875d447bebd`.
Filters: screening within ±30 days of arrest, `is_recid != -1`,
`c_charge_degree != "O"`, `score_text != "N/A"`, decile scores 1–10; African
American (3,175) and Caucasian (2,103). The outcome is `two_year_recid`, a
*recorded* rearrest. ProPublica's article reports FPR 45% vs 23% on a slightly
different subset; these records give 42% vs 22% at cut-off 5.

## CSV upload

```csv
group,y_true,score
A,1,0.8
A,0,0.3
B,1,0.6
B,0,0.2
```

Pick the group, outcome and score columns, two groups, and the outcome value that
means Y = 1. Higher scores must mean Y = 1 is more likely; 0/1 decisions also work.
Uploads with more than 21 distinct scores get 21 quantile cut-offs. UTF-8, at most
8 MB and 200,000 rows. The app offers the COMPAS data as an example download;
`example_upload.csv` is a small example with probability scores. Nothing is
trained. In the browser version the file never leaves the browser.

## Checks

`app.py` ends with a `checks` cell (COMPAS counts, the 5/7 pair on the FPR Pareto
front, Chouldechova's identity, and a CSV round trip). It runs with the app:

```bash
uv run python -m marimo check app.py
uv run python app.py
```

References: [Fairness and Machine Learning](https://fairmlbook.org/classification.html),
Chouldechova (2017), Kleinberg et al. (2016).
