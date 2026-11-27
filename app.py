# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.23.0"]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="Fairness: which error, compared with whom?")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def helpers():
    import csv
    import math
    from bisect import bisect_left
    from io import StringIO

    # ProPublica compas-scores-two-years.csv, standard filters (see README).
    # Per decile score 1..10: (not rearrested, rearrested within two years).
    COMPAS = {
        "Caucasian": [(477, 128), (221, 100), (156, 82), (145, 98), (109, 91),
                      (67, 93), (45, 68), (24, 72), (22, 55), (15, 35)],
        "African American": [(280, 85), (241, 105), (173, 125), (179, 158), (165, 158),
                             (131, 187), (134, 209), (86, 215), (88, 229), (37, 190)],
    }
    GROUP_COLORS = ("#1b9e77", "#7570b3")
    TOL = 0.02  # ponytail: fixed "equal = within 2 pp" rule; add a slider if students need to explore it
    MAX_CUTS = 21  # ponytail: uploads with more distinct scores get 21 quantile cut-offs (21² pairs on the Pareto plot)

    # err: which error is in the numerator; cond: what the rate conditions on (the denominator)
    METRICS = [
        dict(key="sel", crit="Independence", err=None, cond="everyone", num=("tp", "fp"), den=("tp", "fn", "fp", "tn"),
             formula="(TP+FP) / N", eq=r"p(\hat Y = 1 \mid A)"),
        dict(key="fpr", crit="Separation", err="FP", cond="truth", num=("fp",), den=("fp", "tn"),
             formula="FP / (FP+TN)", eq=r"p(\hat Y = 1 \mid Y = 0, A)"),
        dict(key="fnr", crit="Separation", err="FN", cond="truth", num=("fn",), den=("fn", "tp"),
             formula="FN / (FN+TP)", eq=r"p(\hat Y = 0 \mid Y = 1, A)"),
        dict(key="fdr", crit="Sufficiency", err="FP", cond="label", num=("fp",), den=("fp", "tp"),
             formula="FP / (FP+TP)", eq=r"p(Y = 0 \mid \hat Y = 1, A)"),
        dict(key="for_", crit="Sufficiency", err="FN", cond="label", num=("fn",), den=("fn", "tn"),
             formula="FN / (FN+TN)", eq=r"p(Y = 1 \mid \hat Y = 0, A)"),
        dict(key="ba", crit="Performance", err=None, cond=None, num=(), den=(),
             formula="(TPR+TNR) / 2", eq=""),
    ]

    def compas_rows():
        """Expand the counts to one (group, y, score) row per defendant."""
        return [(g, y, s) for g, counts in COMPAS.items() for s, pair in enumerate(counts, 1)
                for y, n in enumerate(pair) for _ in range(n)]

    def build(rows, names, words, kind="compas", intro="", notes="", default=None):
        """Confusion counts per group for every cut-off (flag when score >= cut-off). Both groups share the cut-offs."""
        cuts = sorted({s for _, _, s in rows})
        if len(cuts) < 2:
            raise ValueError("The scores must take at least two different values.")
        if len(cuts) > MAX_CUTS:
            picked = [cuts[round(i * (len(cuts) - 1) / (MAX_CUTS - 1))] for i in range(MAX_CUTS)]
            cuts = sorted({r if (r := float(f"{c:.4g}")) <= c else c for c in picked})
        cm = {}
        for g in names:
            pos = sorted(s for gg, y, s in rows if gg == g and y == 1)
            neg = sorted(s for gg, y, s in rows if gg == g and y == 0)
            cm[g] = {}
            for t in cuts:
                fn, tn = bisect_left(pos, t), bisect_left(neg, t)
                cm[g][t] = dict(tp=len(pos) - fn, fn=fn, fp=len(neg) - tn, tn=tn)
        if default is None:
            default = next((t for t in cuts if t >= 0.5), cuts[-1]) if 0 <= cuts[0] and cuts[-1] <= 1 else cuts[len(cuts) // 2]
        return dict(names=list(names), cuts=cuts, cm=cm, default=default, words=words, kind=kind, intro=intro, notes=notes)

    def rates(c):
        div = lambda a, b: a / b if b else float("nan")
        tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
        r = dict(base=div(tp + fn, tp + fn + fp + tn), sel=div(tp + fp, tp + fn + fp + tn),
                 fpr=div(fp, fp + tn), fnr=div(fn, fn + tp), fdr=div(fp, fp + tp), for_=div(fn, fn + tn))
        r["ba"] = 1 - (r["fpr"] + r["fnr"]) / 2  # = (TPR + TNR) / 2
        return r

    def pair_points(data, metric):
        """One point per pair of cut-offs: gap in `metric` and balanced accuracy of both groups together."""
        a, b = data["names"]
        pts = []
        for ta in data["cuts"]:
            for tb in data["cuts"]:
                ca, cb = data["cm"][a][ta], data["cm"][b][tb]
                gap = abs(rates(ca)[metric] - rates(cb)[metric])
                if gap == gap:  # skip undefined rates
                    pts.append(dict(ta=ta, tb=tb, gap=gap, ba=rates({k: ca[k] + cb[k] for k in ca})["ba"]))
        return pts

    def pareto(pts):
        """Points that no other point beats on both a smaller gap and a higher balanced accuracy."""
        front, best = [], -1.0
        for p in sorted(pts, key=lambda p: (p["gap"], -p["ba"])):
            if p["ba"] > best:
                front.append(p)
                best = p["ba"]
        return front

    def parse_csv(contents):
        if len(contents) > 8_000_000:
            raise ValueError("Use a CSV smaller than 8 MB.")
        try:
            text = contents.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Save the CSV using UTF-8 encoding.") from exc
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(StringIO(text), dialect=dialect)
        headers = reader.fieldnames or []
        if len(headers) < 3 or len(set(headers)) != len(headers) or any(not h for h in headers):
            raise ValueError("Use at least three uniquely named columns: group, outcome and score.")
        rows = []
        try:
            for row in reader:
                if not any(str(v).strip() for v in row.values() if v is not None):
                    continue
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("A CSV row has a different number of fields from the header.")
                rows.append({k: v.strip() for k, v in row.items()})
                if len(rows) > 200_000:
                    raise ValueError("Use at most 200,000 rows.")
        except csv.Error as exc:
            raise ValueError("Could not parse this CSV. Check its delimiters and quoting.") from exc
        if not rows:
            raise ValueError("The CSV has no rows.")
        return headers, rows

    def rows_from_upload(rows, group_col, label_col, score_col, group_a, group_b, positive):
        if not group_a or not group_b or group_a == group_b:
            raise ValueError("Choose two different groups.")
        if len({group_col, label_col, score_col}) != 3:
            raise ValueError("Choose a different column for group, outcome and score.")
        kept = [r for r in rows if r[group_col] in (group_a, group_b)]
        labels = {r[label_col] for r in kept}
        if "" in labels or len(labels) > 2:
            raise ValueError("The outcome must be binary, with no missing values.")
        try:
            out = [(r[group_col], int(r[label_col] == positive), float(r[score_col])) for r in kept]
        except ValueError as exc:
            raise ValueError("Scores must be numbers (probabilities, ranks or 0/1 decisions), with no missing values.") from exc
        if not all(math.isfinite(s) for _, _, s in out):
            raise ValueError("Scores must be finite numbers.")
        if {g for g, _, _ in out} != {group_a, group_b}:
            raise ValueError("Both groups need at least one row.")
        return out, len(rows) - len(kept)

    def example_csv():
        stream = StringIO()
        writer = csv.writer(stream)
        writer.writerow(["group", "y_true", "score"])
        writer.writerows(compas_rows())
        return stream.getvalue().encode()

    return (COMPAS, GROUP_COLORS, METRICS, TOL, build, compas_rows, example_csv,
            pair_points, pareto, parse_csv, rates, rows_from_upload)


@app.cell
def controls(mo):
    source = mo.ui.radio({"COMPAS (ProPublica)": "compas", "Upload your own CSV": "upload"},
                         value="COMPAS (ProPublica)")
    upload = mo.ui.file(filetypes=[".csv"], multiple=False, max_size=8_000_000, label="Choose a CSV")
    numerator = mo.ui.radio({"False positives: flagged, but the outcome did not happen": "FP",
                             "False negatives: not flagged, but the outcome happened": "FN"}, value=None)
    denominator = mo.ui.radio({"Equal chance of being misjudged: compare people with the same true outcome": "truth",
                               "Equal reliability of the label: compare people with the same prediction": "label"},
                              value=None)
    return denominator, numerator, source, upload


@app.cell
def raw_upload(parse_csv, source, upload):
    upload_headers, upload_rows_raw, upload_error = [], [], ""
    if source.value == "upload":
        if not upload.value:
            upload_error = "Choose a CSV file."
        else:
            try:
                upload_headers, upload_rows_raw = parse_csv(upload.value[0].contents)
            except ValueError as _error:
                upload_error = str(_error)
    return upload_error, upload_headers, upload_rows_raw


@app.cell
def column_mapping(mo, upload_headers):
    _headers = upload_headers or ["group", "y_true", "score"]

    def _find(names, fallback):
        return next((h for h in _headers if h.lower() in names), _headers[min(fallback, len(_headers) - 1)])

    group_column = mo.ui.dropdown(_headers, value=_find(["group", "race", "gender", "sex"], 0), label="Group column")
    label_column = mo.ui.dropdown(_headers, value=_find(["y_true", "label", "target", "outcome", "two_year_recid"], 1),
                                  label="Outcome column")
    score_column = mo.ui.dropdown(_headers, value=_find(["score", "y_score", "probability", "decile_score", "y_pred"], 2),
                                  label="Score column")
    return group_column, label_column, score_column


@app.cell
def value_mapping(group_column, label_column, mo, upload_rows_raw):
    _groups = sorted({r.get(group_column.value, "") for r in upload_rows_raw} - {""}) or ["A", "B"]
    _labels = sorted({r.get(label_column.value, "") for r in upload_rows_raw} - {""}) or ["0", "1"]
    group_a = mo.ui.dropdown(_groups, value=_groups[0], label="Group A")
    group_b = mo.ui.dropdown(_groups, value=_groups[1] if len(_groups) > 1 else None,
                             allow_select_none=True, label="Group B")
    positive_label = mo.ui.dropdown(_labels, value="1" if "1" in _labels else _labels[-1], label="Outcome value meaning Y = 1")
    return group_a, group_b, positive_label


@app.cell
def dataset(COMPAS, build, compas_rows, group_a, group_b, group_column, label_column, positive_label,
            rows_from_upload, score_column, source, upload_error, upload_rows_raw):
    data, data_error = None, upload_error
    if source.value == "upload" and upload_rows_raw and not upload_error:
        try:
            _rows, _excluded = rows_from_upload(upload_rows_raw, group_column.value, label_column.value, score_column.value,
                                                group_a.value, group_b.value, positive_label.value)
            data = build(
                _rows, [group_a.value, group_b.value], dict(pos="Y = 1", neg="Y = 0", flag="flagged"), kind="upload",
                intro=(f"**{len(_rows):,} rows** in two groups: {group_a.value} and {group_b.value}. "
                       "**You choose the cut-off:** everyone with a score at or above it is *flagged* (predicted Y = 1)."),
                notes=(f"Y = 1 means `{label_column.value}` = `{positive_label.value}`. Higher scores should mean Y = 1 is "
                       f"more likely. {_excluded:,} rows from other groups were left out. Nothing is trained: the app only "
                       "compares your scores with your outcomes."))
        except ValueError as _error:
            data_error = str(_error)
    if data is None:
        data = build(
            compas_rows(), list(COMPAS), dict(pos="rearrested", neg="not rearrested", flag="flagged high risk"), default=5,
            intro=("COMPAS gives each defendant a risk score from 1 (low) to 10 (high). ProPublica collected the scores "
                   "and two-year rearrest records of 5,278 defendants in Broward County, Florida. "
                   "**You choose the cut-off:** everyone at or above it is flagged *high risk*."),
            notes=("ProPublica's article reports FPR 45% vs 23% using a slightly different subset; these 5,278 records "
                   "give 42% vs 22% at cut-off 5. Same pattern. The outcome is *recorded* rearrest within two years: it "
                   "also reflects who gets policed, so rearrest is not the same as offending. "
                   "[Source](https://github.com/propublica/compas-analysis) (`compas-scores-two-years.csv`, standard filters)."))
    return data, data_error


@app.cell
def cutoffs(GROUP_COLORS, data, mo):
    sliders = mo.ui.dictionary({g: mo.ui.slider(steps=data["cuts"], value=data["default"], show_value=True,
                                                full_width=True, label=f"<span style='color:{c}'>■</span> **{g}**")
                                for g, c in zip(data["names"], GROUP_COLORS)})
    return (sliders,)


@app.cell
def choice(denominator, numerator):
    chosen = {("FP", "truth"): "fpr", ("FN", "truth"): "fnr",
              ("FP", "label"): "fdr", ("FN", "label"): "for_"}.get((numerator.value, denominator.value))
    return (chosen,)


@app.cell
def sidebar(METRICS, chosen, denominator, mo, numerator, sliders, source):
    _spec = next((m for m in METRICS if m["key"] == chosen), None)
    mo.sidebar([
        mo.md("### Data"), source,
        mo.md("### Cut-offs\nFlag everyone with a score at or above:"), sliders.vstack(gap=0.5),
        mo.md("### Step 1 · Numerator\nWhich error matters most?"), numerator,
        mo.md("<small>**Detention** (punitive): a false positive detains someone who would not be "
              "rearrested; a false negative releases someone who is. **Support programme** (assistive): a false positive helps someone "
              "who did not need it; a false negative leaves someone without help.</small>"),
        mo.md("### Step 2 · Denominator\nWhose risk do you compare?"), denominator,
        mo.md(f"**→ You compare {_spec['key'].rstrip('_').upper()}** ({_spec['crit'].lower()}):  \n"
              f"${_spec['eq']}$") if _spec else
        mo.md("*Pick both steps to choose one metric.*"),
    ], width="340px")
    return


@app.cell
def header(data, data_error, example_csv, group_a, group_b, group_column, label_column, mo,
           positive_label, score_column, source, upload):
    _items = [mo.md("# Fairness: which error, compared with whom?\n\n"
                    "*Javier Garcia-Bernardo · Utrecht University & ODISSEI SoDa*"), mo.md(data["intro"])]
    if source.value == "upload":
        _items += [
            mo.md("One row per person: a group, the observed outcome (two values) and a score where higher means "
                  "Y = 1 is more likely (0/1 decisions also work). In the web version the file never leaves your browser."),
            mo.hstack([upload, mo.download(example_csv(), filename="compas_example.csv", label="Example CSV (COMPAS)")],
                      justify="start"),
            mo.hstack([group_column, label_column, score_column], justify="start", wrap=True),
            mo.hstack([group_a, group_b, positive_label], justify="start", wrap=True),
        ]
        if data_error:
            _items.append(mo.callout(mo.md(f"{data_error} Showing COMPAS until this is fixed."), kind="warn"))
    mo.vstack(_items)
    return


@app.cell
def tasks(data, mo):
    mo.callout(mo.md("""
    **Tasks.** Set the cut-offs and steps 1–2 in the sidebar; the matrices and bars below update as you go.

    1. Keep both cut-offs at **5** (ProPublica's "medium or high risk"). On which metrics do the groups
       differ? Is the model fair?
    2. **Step 1 · Numerator.** COMPAS informs detention decisions. Which error matters most? Which bars stay?
    3. **Step 2 · Denominator.** ProPublica compared people with the same true outcome; the vendor compared people
       with the same label. Pick one, then move the cut-offs until the gap closes. What happens to the other bars?
    4. **Step 3 · Pareto.** How much balanced accuracy does closing the gap cost? Is "the same cut-off for
       everyone" a good rule?
    """), kind="info") if data["kind"] == "compas" else None
    return


@app.cell
def dashboard(GROUP_COLORS, METRICS, TOL, chosen, data, denominator, mo, numerator, rates, sliders):
    _a, _b = data["names"]
    _r = {g: rates(data["cm"][g][sliders.value[g]]) for g in data["names"]}
    _w = data["words"]
    _color = dict(zip(data["names"], GROUP_COLORS))
    _spec = next((m for m in METRICS if m["key"] == chosen), None)

    def _swatch(g):
        return (f"<span style='display:inline-block;width:11px;height:11px;border-radius:2px;"
                f"background:{_color[g]};margin-right:6px'></span>")

    def _matrix(g):
        c = data["cm"][g][sliders.value[g]]

        def cell(k, correct):
            bg, fg = ("#00628b", "white") if correct else ("#ff8080", "#222")
            style = f"background:{bg};color:{fg};padding:6px 12px;text-align:center;min-width:72px;border-radius:3px"
            if _spec and k not in _spec["den"]:
                style += ";opacity:0.2"
            if _spec and k in _spec["num"]:
                style += ";outline:3px solid black;outline-offset:-3px"
            return f"<td style='{style}'><small>{k.upper()}</small><br><b>{c[k]:,}</b></td>"

        th = "style='text-align:right;font-weight:normal;padding-right:6px'"
        return (f"<div style='margin-bottom:16px'><div style='margin-bottom:4px'>{_swatch(g)}<b>{g}</b> "
                f"<span style='color:#666'>· score ≥ {sliders.value[g]:g} · {_w['pos']}: {_r[g]['base']:.0%}</span></div>"
                "<table style='border-collapse:separate;border-spacing:3px'>"
                "<tr><th></th><th style='font-weight:normal'>Flagged</th><th style='font-weight:normal'>Not flagged</th></tr>"
                f"<tr><th {th}>{_w['pos'].capitalize()}</th>{cell('tp', True)}{cell('fn', False)}</tr>"
                f"<tr><th {th}>{_w['neg'].capitalize()}</th>{cell('fp', False)}{cell('tn', True)}</tr></table></div>")

    _name = dict(sel="Selection rate", fpr="FPR", fnr="FNR", fdr="FDR", for_="FOR", ba="Balanced accuracy")
    _q = dict(sel=f"Share {_w['flag']}", fpr=f"{_w['neg']}, but {_w['flag']}", fnr=f"{_w['pos']}, but not flagged",
              fdr=f"{_w['flag']}, but {_w['neg']}", for_=f"not flagged, but {_w['pos']}",
              ba="Accuracy, weighting both outcomes equally")

    def _solid(m):
        return m["key"] == "ba" or (numerator.value in (None, m["err"]) and denominator.value in (None, m["cond"]))

    def _bar(g, k):
        v = _r[g][k]
        if v != v:
            return "<div style='margin:3px 0'><small>–</small></div>"
        return (f"<div title='{g}: {v:.1%}' style='display:flex;align-items:center;gap:6px;margin:3px 0'>"
                "<div style='flex:0 0 150px;height:12px;background:#eceff1;border-radius:2px'>"
                f"<div style='width:{100 * v:.1f}%;height:100%;background:{_color[g]};border-radius:0 4px 4px 0'></div>"
                f"</div><small style='font-variant-numeric:tabular-nums'>{v:.{1 if k == 'ba' else 0}%}</small></div>")

    def _gap(k):
        x, y = _r[_a][k], _r[_b][k]
        if x != x or y != y:
            return "–"
        return "✅" if abs(y - x) <= TOL else f"{100 * (y - x):+.{1 if k == 'ba' else 0}f} pp"

    def _row(m):
        k = m["key"]
        style = (f"opacity:{1 if _solid(m) else 0.25};border-top:1px solid #e5e7eb"
                 + (";background:#fff3c4" if k == chosen else ""))
        return (f"<tr style='{style}'>"
                f"<td style='padding:6px 12px 6px 6px'><b>{_name[k]}</b> &nbsp;<code style='font-size:0.8em'>"
                f"{m['formula']}</code><br><small style='color:#555'>{_q[k][0].upper() + _q[k][1:]}</small></td>"
                f"<td style='padding:6px 12px'>{_bar(_a, k)}{_bar(_b, k)}</td>"
                f"<td style='padding:6px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums'>{_gap(k)}</td></tr>")

    def _crit(m):  # one small header row per fairness criterion
        return ("<tr><td colspan='3' style='padding:12px 6px 2px;font-size:0.72em;letter-spacing:0.06em;"
                f"text-transform:uppercase;color:#888'>{m['crit']}</td></tr>")

    _legend = " &nbsp; ".join(f"{_swatch(g)}{g}" for g in data["names"])
    _chart = (f"<p>{_legend}</p><table style='border-collapse:collapse;width:100%'>"
              + "".join((_crit(m) if i == 0 or m["crit"] != METRICS[i - 1]["crit"] else "") + _row(m)
                        for i, m in enumerate(METRICS))
              + f"</table><p><small>Gap = {_b} − {_a}; ✅ = within {100 * TOL:.0f} percentage points. "
              "Balanced accuracy = 1 − (FPR + FNR) / 2.</small></p>")
    _note = (mo.md("<small>**Outlined:** the error you count (numerator). **Faded:** people outside the group you "
                   "compare (denominator).</small>").style({"max-width": "330px"}) if _spec else mo.md(""))

    mo.hstack([
        mo.vstack([mo.md("### Confusion matrices"), mo.Html("".join(_matrix(g) for g in data["names"])), _note]),
        mo.vstack([mo.md("### Metrics per group"), mo.Html(_chart)]),
    ], widths=[1, 1.7], align="start", gap=3)
    return


@app.cell
def pareto_view(chosen, data, mo, pair_points, pareto, sliders):
    _a, _b = data["names"]

    def _ticks(lo, hi, steps):
        step = next((s for s in steps if (hi - lo) / s <= 5), steps[-1])
        return [i * step for i in range(int(lo / step) + 1, int(hi / step) + 1)]

    def _svg(pts, front, same, cur, abbr):
        W, H, L, R, T, B = 640, 380, 64, 16, 16, 48
        # ponytail: zoom to the front and the same-cut-off pairs; pairs with larger gaps are dominated anyway
        xmax = max(p["gap"] for p in front + same + [cur] if p) * 1.1 or 0.01
        pts = [p for p in pts if p["gap"] <= xmax]
        lo, hi = min(p["ba"] for p in pts), max(p["ba"] for p in pts)
        pad = (hi - lo) * 0.06 or 0.01
        lo, hi = lo - pad, hi + pad
        X = lambda v: L + v / xmax * (W - L - R)
        Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)

        def mark(p, r, fill, extra=""):
            return f"<circle cx='{X(p['gap']):.1f}' cy='{Y(p['ba']):.1f}' r='{r}' fill='{fill}' {extra}/>"

        def line(ps, color):
            xy = " ".join(f"{X(p['gap']):.1f},{Y(p['ba']):.1f}" for p in ps)
            return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{xy}'/>"

        def hover(p):
            # invisible hit area + a tooltip that CSS shows on hover; drawn last so it sits on top
            x, y = X(p["gap"]), Y(p["ba"])
            rows = [f"{_a} ≥ {p['ta']:g}, {_b} ≥ {p['tb']:g}",
                    f"gap {100 * p['gap']:.1f} pp · balanced accuracy {p['ba']:.3f}"]
            w = 6.3 * max(map(len, rows)) + 14
            tx = x + 10 if x + 10 + w < W else x - 10 - w
            ty = y - 48 if y - 48 > 0 else y + 10
            text = "".join(f"<text x='{tx + 7:.1f}' y='{ty + 16 + 15 * i:.1f}' font-size='11'>{s}</text>"
                           for i, s in enumerate(rows))
            return (f"<g class='pt'><circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='transparent'/><g class='tip'>"
                    f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='none' stroke='black' stroke-width='1.5'/>"
                    f"<rect x='{tx:.1f}' y='{ty:.1f}' width='{w:.0f}' height='38' rx='4' fill='white' stroke='#888'/>"
                    f"{text}</g></g>")

        labels, placed = "", []
        for p in front:  # cut-off pair next to each front point, skipping labels that would overlap
            x, y = X(p["gap"]) + 7, Y(p["ba"]) - 6
            anchor = "start"
            if x > W - 60:
                x, anchor = x - 14, "end"
            if all(abs(x - px) > 36 or abs(y - py) > 11 for px, py in placed):
                labels += (f"<text x='{x:.1f}' y='{y:.1f}' font-size='10' font-weight='bold' fill='#2b6795' "
                           f"stroke='white' stroke-width='3' paint-order='stroke' text-anchor='{anchor}'>"
                           f"{p['ta']:g}/{p['tb']:g}</text>")
                placed.append((x, y))

        ticks = "".join(
            f"<line x1='{X(v):.1f}' x2='{X(v):.1f}' y1='{H - B}' y2='{H - B + 5}' stroke='#555'/>"
            f"<text x='{X(v):.1f}' y='{H - B + 18}' text-anchor='middle' font-size='11'>{100 * v:.0f}</text>"
            for v in [0] + _ticks(0, xmax, [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5])) + "".join(
            f"<line x1='{L - 5}' x2='{L}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='#555'/>"
            f"<text x='{L - 8}' y='{Y(v) + 4:.1f}' text-anchor='end' font-size='11'>{v:.2f}</text>"
            for v in _ticks(lo, hi, [0.01, 0.02, 0.05, 0.1, 0.2]))
        axes = (f"<line x1='{L}' x2='{W - R}' y1='{H - B}' y2='{H - B}' stroke='#555'/>"
                f"<line x1='{L}' x2='{L}' y1='{T}' y2='{H - B}' stroke='#555'/>"
                f"<text x='{(L + W - R) / 2}' y='{H - 8}' text-anchor='middle' font-size='12'>"
                f"Gap in {abbr} between the groups (percentage points)</text>"
                f"<text transform='translate(14,{(T + H - B) / 2}) rotate(-90)' text-anchor='middle' font-size='12'>"
                "Balanced accuracy (both groups)</text>")
        body = ("".join(mark(p, 3.5, "#c8c8c8") for p in pts)
                + line(same, "#e08214") + "".join(mark(p, 4.5, "#e08214") for p in same)
                + line(front, "#2b6795") + "".join(mark(p, 4.5, "#2b6795") for p in front) + labels
                + (mark(cur, 9, "none", "stroke='black' stroke-width='2.5'") if cur else "")
                + "".join(hover(p) for p in pts))
        return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='max-width:{W}px;font-family:sans-serif'>"
                "<style>.pt .tip{visibility:hidden}.pt:hover .tip{visibility:visible}</style>"
                f"{axes}{ticks}{body}</svg>")

    _items = [mo.md("## Step 3 · What does closing the gap cost?")]
    if chosen is None:
        _items.append(mo.md("*Choose steps 1 and 2 in the sidebar first.*"))
    else:
        _abbr = chosen.rstrip("_").upper()
        _pts = pair_points(data, chosen)
        _same = [p for p in _pts if p["ta"] == p["tb"]]
        _cur = next((p for p in _pts if (p["ta"], p["tb"]) == (sliders.value[_a], sliders.value[_b])), None)
        _legend = " &nbsp; ".join(
            f"<span style='color:{c};font-size:18px'>●</span> {label}" for c, label in [
                ("#c8c8c8", "one pair of cut-offs"), ("#e08214", "same cut-off for both groups"),
                ("#2b6795", f"Pareto front, labelled {_a} / {_b} cut-off"), ("black", "◯ your sliders")])
        _items += [
            mo.md(f"Each dot is one pair of cut-offs. Left means a smaller gap in {_abbr}; up means more accurate. "
                  "The **Pareto front** holds the pairs where you cannot shrink the gap without losing accuracy. "
                  "Hover over a dot to see its cut-offs. The black ring follows the sliders in the sidebar."),
            mo.Html(f"<p>{_legend}</p>" + _svg(_pts, pareto(_pts), _same, _cur, _abbr)),
        ]
    mo.vstack(_items)
    return


@app.cell
def debrief(data, mo, rates, sliders):
    def _row(g):
        r = rates(data["cm"][g][sliders.value[g]])
        try:
            implied = r["base"] / (1 - r["base"]) * r["fdr"] / (1 - r["fdr"]) * (1 - r["fnr"])
        except ZeroDivisionError:
            implied = float("nan")
        return (f"| {g} (≥ {sliders.value[g]:g}) | {r['base']:.2f} | {r['fdr']:.2f} | {r['fnr']:.2f} "
                f"| **{implied:.2f}** | {r['fpr']:.2f} |")

    _p = " vs ".join(f"{rates(data['cm'][g][data['cuts'][0]])['base']:.0%}" for g in data["names"])
    mo.accordion({
        "Why can't you have it all? (open after the exercise)": mo.vstack([
            mo.md(r"""
            Within each group, with base rate $p$ (share with $Y = 1$), the three error rates are tied
            together (Chouldechova, 2017; she writes PPV = 1 − FDR):

            $$\text{FPR} = \frac{p}{1-p}\cdot\frac{\text{FDR}}{1-\text{FDR}}\cdot(1-\text{FNR})$$

            Check it with your cut-offs:
            """),
            mo.md("\n".join(["| Group (cut-off) | $p$ | FDR | FNR | Formula gives FPR | Actual FPR |",
                             "|---|--:|--:|--:|--:|--:|"] + [_row(g) for g in data["names"]])),
            mo.md(
                f"The base rates differ ({_p}) and the model is not perfect. So if FPR **and** FNR are equal "
                "(separation), the formula forces FDR to differ (no sufficiency). And if FDR and FNR are equal, "
                "FPR must differ. Only equal base rates or a perfect model escape this. **You need to choose.**"),
        ]),
        "About the data": mo.md(data["notes"]),
    })
    return


@app.cell
def checks(COMPAS, build, compas_rows, example_csv, pair_points, pareto, parse_csv, rates, rows_from_upload):
    # ponytail: one smoke test of the numbers the lecture relies on, plus the CSV upload path
    _w = dict(pos="rearrested", neg="not rearrested", flag="flagged high risk")
    _d = build(compas_rows(), list(COMPAS), _w, default=5)
    assert sum(sum(_d["cm"][g][1].values()) for g in COMPAS) == 5278
    assert _d["cm"]["African American"][5] == dict(tp=1188, fn=473, fp=641, tn=873)
    assert any((p["ta"], p["tb"]) == (5, 7) for p in pareto(pair_points(_d, "fpr")))
    for _g in COMPAS:
        for _t in _d["cuts"]:  # Chouldechova (2017): FPR = p/(1-p) * FDR/(1-FDR) * (1-FNR)
            _r = rates(_d["cm"][_g][_t])
            assert abs(_r["base"] / (1 - _r["base"]) * _r["fdr"] / (1 - _r["fdr"]) * (1 - _r["fnr"]) - _r["fpr"]) < 1e-9
    _rows, _ = rows_from_upload(parse_csv(example_csv())[1], "group", "y_true", "score",
                                "Caucasian", "African American", "1")
    assert build(_rows, list(COMPAS), _w)["cm"] == _d["cm"]
    return


if __name__ == "__main__":
    app.run()
