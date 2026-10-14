from __future__ import annotations

import io
import json
import math
import re
import time
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns continued."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in consumer prices."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, reported in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP."),
}
WB_ROOT = "https://api.worldbank.org/v2"
CONTEXTS = {
    "Business": ("Revenue", "currency units", "time", "years", "100 + 12*x - 0.4*x**2", "A business indicator can represent revenue or demand over time."),
    "Engineering": ("Displacement", "meters", "time", "seconds", "2*x**2 - 3*x + 1", "A displacement model can describe position along a line."),
    "Medical": ("Concentration", "mg/L", "time", "hours", "8*exp(-0.25*x)", "A concentration model can represent a simplified decay process, not medical advice."),
    "Sports science": ("Performance", "score units", "training time", "hours", "10 + 3*x - 0.2*x**2", "A performance score may rise and then plateau or decline."),
    "Biology": ("Population", "organisms", "time", "days", "100*exp(0.15*x)", "An exponential model can represent idealized growth over a limited interval."),
    "Bioengineering": ("Signal", "signal units", "time", "seconds", "sin(x) + 0.2*x", "A signal model can combine periodic and trend behavior."),
    "Economic forecasting": ("Index", "index points", "time", "years", "100*exp(0.03*x)", "An index model can describe a simplified growth trend, not a guaranteed forecast."),
    "Weather": ("Temperature", "°C", "time", "hours", "15 + 8*sin(2*pi*x/24)", "A sinusoid can approximate a repeating daily temperature pattern."),
    "Custom": ("Output", "units", "Input", "units", "x**2 - 4*x + 3", "Describe your own context and interpret the output carefully."),
}
SAFE_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
    "abs": sp.Abs, "pi": sp.pi, "E": sp.E, "sinh": sp.sinh, "cosh": sp.cosh,
}
x = sp.Symbol("x", real=True)

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("📊 " + APP_TITLE)
st.caption("Authentic World Bank indicators + editable data + polynomial modeling + function and calculus lab")
st.info("Research question: add your team's question in the sidebar. Select your featured countries, indicator, and mathematics pathway before interpreting results.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    """Fetch one World Bank indicator; never fabricate missing observations."""
    url = f"{WB_ROOT}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Network/API error: {exc}"
    except (ValueError, json.JSONDecodeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not read API response: {exc}"
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        message = payload[0].get("message", "No data returned") if isinstance(payload, list) and payload and isinstance(payload[0], dict) else "Unexpected or empty World Bank response"
        return pd.DataFrame(columns=["year", "value"]), str(message)
    rows = []
    for item in payload[1]:
        if not isinstance(item, dict) or item.get("date") is None:
            continue
        try:
            yr = int(item["date"])
            val = item.get("value")
            if val is not None and np.isfinite(float(val)):
                rows.append({"year": yr, "value": float(val)})
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(rows, columns=["year", "value"])
    if not df.empty:
        df = df.drop_duplicates("year", keep="last").sort_values("year").reset_index(drop=True)
    return df, ""

def safe_expression(text: str) -> sp.Expr:
    """Parse familiar math notation while allowing only a small symbol/function whitelist."""
    cleaned = text.strip().replace("^", "**")
    if not cleaned or len(cleaned) > 180:
        raise ValueError("Enter a function no longer than 180 characters.")
    # Restrict the character set and identifiers before parsing user input.
    if not re.fullmatch(r"[A-Za-z0-9_+*/().,\-\s*]+", cleaned):
        raise ValueError("Use only numbers, x, parentheses, arithmetic operators, and approved function names.")
    identifiers = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", cleaned)
    allowed_names = set(SAFE_FUNCS) | {"x"}
    unknown_names = sorted(set(identifiers) - allowed_names)
    if unknown_names:
        raise ValueError("Unknown name(s): " + ", ".join(unknown_names) + ". Use x and approved functions such as sin, cos, exp, log, and sqrt.")
    try:
        expr = sp.sympify(cleaned, locals={**SAFE_FUNCS, "x": x}, evaluate=True)
    except Exception as exc:
        raise ValueError("I couldn't parse that function. Try notation such as x**2, sin(x), exp(-x), or 1/(x-2).") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Use x as the only variable.")
    if expr.has(sp.Integral, sp.Derivative, sp.Sum, sp.Product):
        raise ValueError("Integrals, derivatives, sums, and products are not accepted as input.")
    if len(str(expr)) > 500:
        raise ValueError("That expression is too complex. Try a simpler function.")
    return expr

def numeric_values(expr: sp.Expr, xs: np.ndarray) -> np.ndarray:
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        ys = np.asarray(fn(xs), dtype=float)
    if ys.ndim == 0:
        ys = np.full(xs.shape, float(ys))
    return np.broadcast_to(ys, xs.shape).astype(float)

def fmt_num(v: Any, digits: int = 5) -> str:
    try:
        v = float(v)
        if not np.isfinite(v):
            return "undefined"
        return f"{v:.{digits}g}"
    except (TypeError, ValueError):
        return str(v)

def validate_working_data(df: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    result = df.copy()
    if not {"year", "value"}.issubset(result.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    years_numeric = pd.to_numeric(result["year"], errors="coerce")
    values_numeric = pd.to_numeric(result["value"], errors="coerce")
    if years_numeric.isna().any() or ((years_numeric.dropna() % 1) != 0).any():
        errors.append("Every year must be a whole-number integer.")
    if years_numeric.duplicated().any():
        errors.append("Years must be unique; remove duplicate years before fitting.")
    if values_numeric.isna().any():
        errors.append("Values must be numeric. Blank values are not accepted in the working table.")
    if np.isinf(values_numeric.dropna().astype(float)).any():
        errors.append("Values must be finite numbers.")
    if not errors:
        result["year"] = years_numeric.astype(int)
        result["value"] = values_numeric.astype(float)
        code = INDICATORS[indicator_name][0]
        if code == "SP.DYN.LE00.IN" and ((result.value < 0) | (result.value > 125)).any():
            errors.append("Life expectancy must be between 0 and 125 years.")
        if code in {"SL.UEM.TOTL.NE.ZS", "GC.DOD.TOTL.GD.ZS"} and (result.value < 0).any():
            errors.append(f"{indicator_name} values should not be negative.")
        if code == "SP.DYN.CBRT.IN" and (result.value < 0).any():
            errors.append("Crude birth rate cannot be negative.")
        result = result.sort_values("year").reset_index(drop=True)
    return result, errors

def fit_polynomial(df: pd.DataFrame, degree: int):
    clean = df.dropna(subset=["year", "value"]).copy()
    clean = clean.sort_values("year")
    if clean.year.nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; only {clean.year.nunique()} are available.")
    base_year = int(clean.year.min())
    t = clean.year.to_numpy(dtype=float) - base_year
    coeff = np.polynomial.polynomial.polyfit(t, clean.value.to_numpy(dtype=float), degree)
    predicted = np.polynomial.polynomial.polyval(t, coeff)
    residuals = clean.value.to_numpy(dtype=float) - predicted
    return {"base_year": base_year, "degree": degree, "coeff": coeff, "predicted": predicted,
            "residuals": residuals, "r2": float(r2_score(clean.value, predicted)) if len(clean) > 1 else float("nan"),
            "mae": float(mean_absolute_error(clean.value, predicted)), "mean": float(clean.value.mean()),
            "n": int(len(clean)), "min_year": int(clean.year.min()), "max_year": int(clean.year.max()),
            "working": clean}

def poly_predict(model: dict, years: np.ndarray) -> np.ndarray:
    return np.polynomial.polynomial.polyval(np.asarray(years, dtype=float) - model["base_year"], model["coeff"])

def equation_text(model: dict) -> str:
    terms = []
    for power, coef in enumerate(model["coeff"]):
        if abs(coef) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{coef:.8g}")
        elif power == 1:
            terms.append(f"{coef:+.8g}t")
        else:
            terms.append(f"{coef:+.8g}t^{power}")
    return "f(t) = " + (" ".join(terms) if terms else "0") + f", where t = year − {model['base_year']}"

def approximate_real_roots(expr: sp.Expr, low: float, high: float, count: int = 1200) -> list[float]:
    if high <= low:
        return []
    grid = np.linspace(low, high, count)
    try:
        vals = numeric_values(expr, grid)
    except Exception:
        return []
    roots = []
    for i in range(len(grid)-1):
        a, b, fa, fb = grid[i], grid[i+1], vals[i], vals[i+1]
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if abs(fa) < 1e-7:
            roots.append(float(a))
        elif fa * fb < 0:
            lo, hi = float(a), float(b)
            for _ in range(35):
                mid = (lo + hi) / 2
                try:
                    fm = float(numeric_values(expr, np.array([mid]))[0])
                except Exception:
                    break
                if not np.isfinite(fm):
                    break
                if abs(fm) < 1e-10:
                    lo = hi = mid
                    break
                if fa * fm <= 0:
                    hi, fb = mid, fm
                else:
                    lo, fa = mid, fm
            roots.append((lo+hi)/2)
    unique = []
    for r in roots:
        if low - 1e-5 <= r <= high + 1e-5 and all(abs(r-u) > 1e-3 for u in unique):
            unique.append(r)
    return unique

def derivative_analysis(expr: sp.Expr, low: float, high: float) -> dict:
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    crit = approximate_real_roots(d1, low, high)
    inflect_candidates = approximate_real_roots(d2, low, high)
    def val(e, v):
        try:
            out = complex(sp.N(e.subs(x, v)))
            return float(out.real) if abs(out.imag) < 1e-8 and math.isfinite(out.real) else float("nan")
        except Exception:
            return float("nan")
    extrema = []
    for c in crit:
        eps = max(1e-4, (high-low)*1e-5)
        left, right = val(d1, max(low, c-eps)), val(d1, min(high, c+eps))
        kind = "local maximum" if left > 0 and right < 0 else "local minimum" if left < 0 and right > 0 else "stationary point, not classified as an extremum"
        extrema.append((c, val(expr,c), kind))
    inflections = []
    for c in inflect_candidates:
        eps = max(1e-4, (high-low)*1e-5)
        left, right = val(d2, max(low,c-eps)), val(d2,min(high,c+eps))
        if np.isfinite(left) and np.isfinite(right) and left*right < 0:
            inflections.append((c,val(expr,c)))
    grid = np.linspace(low, high, 500)
    y = numeric_values(expr, grid)
    finite = np.isfinite(y)
    if finite.any():
        ymin, ymax = float(np.min(y[finite])), float(np.max(y[finite]))
        y_at_low, y_at_high = val(expr, low), val(expr, high)
        candidates = [(low, y_at_low), (high, y_at_high)] + [(c, val(expr,c)) for c in crit]
        candidates = [(a,b) for a,b in candidates if np.isfinite(b)]
        absolute_min = min(candidates, key=lambda z:z[1]) if candidates else None
        absolute_max = max(candidates, key=lambda z:z[1]) if candidates else None
    else:
        ymin = ymax = float("nan")
        absolute_min = absolute_max = None
    return {"d1": d1, "d2": d2, "critical": extrema, "inflections": inflections,
            "ymin": ymin, "ymax": ymax, "absolute_min": absolute_min, "absolute_max": absolute_max}

def chart_figure(data: pd.DataFrame, model: dict, degree: int, sample_step: int,
                 calc_year: int | None, batch_df: pd.DataFrame | None, title: str, unit: str):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(data.year, data.value, label="Observed / working data", color="#1f77b4", s=26, zorder=4)
    left, right = model["min_year"], model["max_year"]
    horizon = int(st.session_state.get("horizon", 0))
    end = min(2100, right + horizon)
    curve_x = np.linspace(left, max(left+1, end), max(350, (end-left+1)*3))
    curve_y = poly_predict(model, curve_x)
    in_mask = curve_x <= right
    ax.plot(curve_x[in_mask], curve_y[in_mask], label="Polynomial fit (historical range)", color="#d62728", linewidth=2)
    if (~in_mask).any():
        ax.plot(curve_x[~in_mask], curve_y[~in_mask], label="Extrapolation (uncertain)", color="#ff7f0e", linestyle="--", linewidth=2)
    sample_years = np.arange(left, right+1, sample_step)
    ax.scatter(sample_years, poly_predict(model, sample_years), label=f"Fit markers every {sample_step} year(s)", marker="x", color="#2ca02c", s=32)
    ax.axvline(right, color="gray", linestyle=":", label=f"Historical boundary: {right}")
    if calc_year is not None:
        py = float(poly_predict(model, np.array([calc_year]))[0])
        ax.scatter([calc_year], [py], color="black", marker="D", s=55, label=f"Single estimate ({calc_year})", zorder=5)
    if batch_df is not None and not batch_df.empty:
        ax.scatter(batch_df["year"], batch_df["estimate"], color="#9467bd", marker="^", s=48, label="Batch estimates", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig

def build_pdf(meta: dict, fig, data: pd.DataFrame, results: pd.DataFrame, model: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=32, leftMargin=32, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]), Spacer(1, 8)]
    for key, value in meta.items():
        story.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("<b>Model</b>", styles["Heading2"]),
              Paragraph(equation_text(model), styles["BodyText"]),
              Paragraph(f"In-sample R² = {fmt_num(model['r2'])}; in-sample MAE = {fmt_num(model['mae'])}; observations = {model['n']}; historical range = {model['min_year']}–{model['max_year']}.", styles["BodyText"]),
              Paragraph("Limitations: polynomial fits describe patterns in the supplied observations and may behave implausibly outside the fitted years. MAE and R² are in-sample metrics, not guarantees or prediction intervals.", styles["BodyText"]),
              Spacer(1, 8)]
    image_buf = io.BytesIO()
    fig.savefig(image_buf, format="png", dpi=130, bbox_inches="tight")
    image_buf.seek(0)
    story.append(RLImage(image_buf, width=690, height=355))
    story.append(Paragraph("Working data preview", styles["Heading2"]))
    preview = data.head(18).copy()
    rows = [["Year", "Value"]] + [[str(int(r.year)), fmt_num(r.value)] for r in preview.itertuples()]
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbeafe")),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(table)
    if results is not None and not results.empty:
        story.append(Paragraph("Calculated-year results", styles["Heading2"]))
        rows = [list(results.columns)] + [[str(v) for v in row] for row in results.head(30).itertuples(index=False, name=None)]
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbeafe")),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(table)
    doc.build(story)
    return buf.getvalue()

with st.sidebar:
    st.header("Project setup")
    research_question = st.text_input("Research question", "How has life expectancy changed over time in selected Latin American countries?")
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    country_code = COUNTRIES[country_label]
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10, key="horizon")
    compare_labels = st.multiselect("Compare with countries", [c for c in COUNTRIES if c != country_label], default=[])
    printer_mode = st.checkbox("Printer-friendly report view", False)
    st.caption("Brazil is included as a Latin American comparison country; it is not a Spanish-speaking country.")

indicator_code, unit, definition = INDICATORS[indicator_name]
current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
retrieval_date = date.today().isoformat()
st.write(f"**Research question:** {research_question}")
st.write(f"**Source:** World Bank API · **Retrieved:** {retrieval_date}")
st.write(f"**Indicator:** {indicator_name} (`{indicator_code}`) · **Units:** {unit}")
st.caption(definition + " Source: https://data.worldbank.org/indicator/" + indicator_code)
with st.spinner("Retrieving official World Bank data…"):
    source_df, api_error = fetch_world_bank(country_code, indicator_code, start_year, end_year)
if api_error:
    st.error("The World Bank request did not succeed. No substitute or random data has been generated. " + api_error)
if source_df.empty:
    st.warning("No non-null observations were returned for this country/indicator and period. Try another indicator or country, or check the API later.")
    st.stop()

st.caption(f"Requested period: {start_year}–{end_year}. Actual coverage: {int(source_df.year.min())}–{int(source_df.year.max())}; {len(source_df)} non-null annual observations.")
st.caption("Source data is kept separate from the editable working table below.")

if "selection_key" not in st.session_state:
    st.session_state.selection_key = None
selection_key = (country_code, indicator_code)
if st.session_state.selection_key != selection_key:
    st.session_state.working_df = source_df.copy()
    st.session_state.source_snapshot = source_df.copy()
    st.session_state.selection_key = selection_key
    st.session_state.single_result = None
    st.session_state.batch_result = None
    st.session_state.result_key = None
    st.session_state.edited = False

st.subheader("1. Source data and editable working table")
st.dataframe(source_df, use_container_width=True, hide_index=True)
edited_df = st.data_editor(st.session_state.working_df, num_rows="dynamic", use_container_width=True, hide_index=True,
                           column_config={"year": st.column_config.NumberColumn("Year", step=1, format="%d"),
                                          "value": st.column_config.NumberColumn(f"Value ({unit})", format="%.6f")}, key="working_editor")
valid_df, validation_errors = validate_working_data(edited_df, indicator_name)
if validation_errors:
    for err in validation_errors:
        st.error(err)
else:
    st.session_state.working_df = valid_df.copy()
    st.session_state.edited = not valid_df.reset_index(drop=True).equals(st.session_state.source_snapshot.reset_index(drop=True))
    if st.session_state.edited:
        st.warning("User-modified scenario: model and exports use your edited working data, not only the original source.")
if st.button("Reset to source"):
    st.session_state.working_df = st.session_state.source_snapshot.copy()
    st.session_state.edited = False
    st.session_state.single_result = None
    st.session_state.batch_result = None
    st.session_state.result_key = None
    st.rerun()

if validation_errors:
    st.stop()
working_df = st.session_state.working_df.copy()
if len(working_df) < degree + 1:
    st.error(f"Degree {degree} needs at least {degree+1} distinct valid years. The working table currently has {working_df.year.nunique()}. Lower the degree or add valid observations.")
    st.stop()
try:
    model = fit_polynomial(working_df, degree)
except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(f"Could not fit model: {exc}")
    st.stop()

st.subheader("2. Polynomial fit")
st.latex(r"t = \text{year} - " + str(model["base_year"]))
st.code(equation_text(model), language="text")
cols = st.columns(5)
cols[0].metric("Observations", model["n"])
cols[1].metric("In-sample R²", fmt_num(model["r2"]))
cols[2].metric("In-sample MAE", fmt_num(model["mae"]) + f" {unit}")
cols[3].metric("Mean observed value", fmt_num(model["mean"]) + f" {unit}")
cols[4].metric("Historical range", f"{model['min_year']}–{model['max_year']}")
residual_df = model["working"][["year", "value"]].copy()
residual_df["fitted_value"] = model["predicted"]
residual_df["residual (actual - fitted)"] = model["residuals"]
with st.expander("Residuals and in-sample fitted values"):
    st.dataframe(residual_df, use_container_width=True, hide_index=True)
    st.caption("A fitted value at an observed year is a model estimate, not the actual observation. Residual = actual value − fitted value.")

# Chronological holdout: fit on earlier years, assess later years only when enough points exist.
holdout_n = max(1, int(round(len(working_df) * .2)))
if len(working_df) >= max(degree + 3, 10) and len(working_df) - holdout_n >= degree + 1:
    train = working_df.iloc[:-holdout_n]
    test = working_df.iloc[-holdout_n:]
    try:
        holdout_model = fit_polynomial(train, degree)
        holdout_pred = poly_predict(holdout_model, test.year.to_numpy())
        st.write(f"**Chronological holdout MAE:** {fmt_num(mean_absolute_error(test.value, holdout_pred))} {unit} across {len(test)} later-year observations. This is separate from in-sample fit metrics.")
    except Exception:
        st.caption("Chronological holdout score unavailable for this data/degree combination.")
else:
    st.caption("Chronological holdout comparison needs more observations than this selection currently provides.")

st.subheader("3. Calculate estimates")
model_key = (country_code, indicator_code, degree, tuple(map(tuple, working_df[["year","value"]].to_numpy().tolist())))
if st.session_state.get("result_key") != model_key:
    st.session_state.single_result = None
    st.session_state.batch_result = None
    st.session_state.result_key = model_key
single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(model["max_year"],1950),2100), step=1)
if st.button("Calculate Value"):
    estimate = float(poly_predict(model, np.array([single_year]))[0])
    is_observed = bool((working_df.year == int(single_year)).any())
    actual = float(working_df.loc[working_df.year == int(single_year), "value"].iloc[0]) if is_observed else None
    st.session_state.single_result = {"year": int(single_year), "estimate": estimate, "classification": "within fitted range" if model["min_year"] <= single_year <= model["max_year"] else "outside fitted range", "is_observed": is_observed, "actual": actual}
if st.session_state.single_result:
    r = st.session_state.single_result
    st.write(f"**Year:** {r['year']} · **Model estimate:** {fmt_num(r['estimate'])} {unit} · **Classification:** {r['classification']}")
    if r["is_observed"]:
        st.write(f"An actual working-data observation exists for this year: {fmt_num(r['actual'])} {unit}. The estimate above is the fitted value, not the observed value.")
    if r["classification"] == "outside fitted range":
        st.warning("Polynomial extrapolation can become implausible quickly. This estimate is not a validated forecast or a prediction interval.")

st.markdown("**Batch years**")
batch_mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
batch_df = None
if batch_mode == "Year range":
    c1,c2,c3 = st.columns(3)
    batch_start = c1.number_input("Start year", 1950, 2100, max(1950, model["max_year"]-5), key="batch_start")
    batch_end = c2.number_input("End year", 1950, 2100, min(2100, model["max_year"]+10), key="batch_end")
    batch_step = c3.slider("Range step", 1, 10, 1)
    requested = list(range(int(batch_start), int(batch_end)+1, int(batch_step))) if batch_end >= batch_start else []
    if batch_end < batch_start:
        st.error("End year must be greater than or equal to start year.")
else:
    custom_text = st.text_input("Comma-separated years", f"{model['min_year']}, {model['max_year']}, {min(2100, model['max_year']+5)}")
    requested, invalid_tokens = [], []
    for token in custom_text.split(","):
        token = token.strip()
        if not token:
            continue
        if not re.fullmatch(r"\d{4}", token) or not 1950 <= int(token) <= 2100:
            invalid_tokens.append(token)
        else:
            requested.append(int(token))
    if invalid_tokens:
        st.warning("Ignored invalid year input(s): " + ", ".join(invalid_tokens) + ". Use whole years from 1950 through 2100.")
requested = list(dict.fromkeys(requested))
if len(requested) > 500:
    st.error("Please request no more than 500 years at once.")
    requested = []
if st.button("Calculate batch years"):
    if not requested:
        st.error("Enter at least one valid year.")
    else:
        vals = poly_predict(model, np.array(requested, dtype=float))
        batch = pd.DataFrame({"year": requested, "estimate": vals})
        batch["units"] = unit
        batch["range_status"] = np.where((batch.year >= model["min_year"]) & (batch.year <= model["max_year"]), "within fitted range", "outside fitted range")
        batch["observed_year"] = batch.year.isin(working_df.year)
        st.session_state.batch_result = batch
if st.session_state.get("batch_result") is not None:
    batch_df = st.session_state.batch_result.copy()
    st.dataframe(batch_df, use_container_width=True, hide_index=True)
    if (batch_df.range_status == "outside fitted range").any():
        st.warning("Some batch estimates are outside the fitted historical range. Treat them as uncertain extrapolations.")
if st.session_state.single_result and st.session_state.batch_result is not None:
    yr = st.session_state.single_result["year"]
    if yr in set(st.session_state.batch_result.year):
        bval = float(st.session_state.batch_result.loc[st.session_state.batch_result.year == yr, "estimate"].iloc[0])
        st.caption(f"Single/batch consistency check for {yr}: difference = {fmt_num(bval - st.session_state.single_result['estimate'], 3)} (should be 0 apart from floating-point rounding).")

st.subheader("4. Historical graph")
fig = chart_figure(working_df, model, degree, sample_step,
                   st.session_state.single_result["year"] if st.session_state.single_result else None,
                   batch_df, f"{country_label} — {indicator_name}", unit)
st.pyplot(fig, use_container_width=True)
with st.expander("Interpret the graph"):
    st.write(f"The points are the {len(working_df)} working observations. The solid curve is the degree-{degree} polynomial fit over {model['min_year']}–{model['max_year']}. The dashed segment is extrapolation, not observed data. Graph sample increment is {sample_step} year(s); changing it changes the X markers, not the smooth curve.")
    st.write(f"In-sample residuals summarize how closely the model follows known data. MAE is the average absolute in-sample error ({fmt_num(model['mae'])} {unit}); R² describes in-sample fit and is not proof of causation.")
    if st.session_state.single_result and not np.isfinite(st.session_state.single_result["estimate"]):
        st.warning("The polynomial estimate is not finite for this input; inspect the selected degree and data.")

st.subheader("5. Compare countries")
if compare_labels:
    comp_rows = []
    comp_fig, comp_ax = plt.subplots(figsize=(10,5))
    for label in compare_labels:
        df_c, err_c = fetch_world_bank(COUNTRIES[label], indicator_code, start_year, end_year)
        if err_c:
            st.warning(f"{label}: {err_c}")
        if df_c.empty:
            st.caption(f"{label}: no non-null observations returned.")
            continue
        comp_rows.append({"Country": label, "First year": int(df_c.year.min()), "Last year": int(df_c.year.max()), "Observations": len(df_c)})
        comp_ax.plot(df_c.year, df_c.value, marker=".", linewidth=1.3, label=label)
    comp_ax.plot(source_df.year, source_df.value, marker=".", linewidth=1.8, label=country_label)
    comp_ax.set_title(f"Country comparison — {indicator_name}")
    comp_ax.set_xlabel("Year"); comp_ax.set_ylabel(unit); comp_ax.grid(True, alpha=.25); comp_ax.legend(fontsize=8); comp_fig.tight_layout()
    st.pyplot(comp_fig, use_container_width=True)
    if comp_rows:
        st.dataframe(pd.DataFrame([{"Country": country_label, "First year": int(source_df.year.min()), "Last year": int(source_df.year.max()), "Observations": len(source_df)}] + comp_rows), use_container_width=True, hide_index=True)
    st.caption("Countries may have different coverage and gaps. Lines connect available annual observations, so compare overlapping years and do not assume missing years are zero.")
else:
    st.caption("Choose one or more comparison countries in the sidebar to display their authentic World Bank observations.")

st.subheader("6. Function builder and calculus lab")
context = st.selectbox("Context", list(CONTEXTS))
default_dep, default_dep_unit, default_ind, default_ind_unit, suggested, context_note = CONTEXTS[context]
function_source = st.radio("Function source", ["Curve of best fit", "Suggested function", "Custom function"], horizontal=True)
if function_source == "Curve of best fit":
    # Translate the fitted polynomial to an x-based expression where x means year.
    t_symbol = sp.Symbol("x", real=True)
    expr = sum(sp.Float(float(c), 12) * (t_symbol - model["base_year"])**i for i,c in enumerate(model["coeff"]))
    expr = sp.expand(expr)
    expr_error = None
    function_label = "Polynomial curve of best fit, with x = calendar year"
elif function_source == "Suggested function":
    suggested_text = st.text_input("Suggested function (you may edit it)", suggested)
    try:
        expr = safe_expression(suggested_text)
        expr_error = None
    except ValueError as exc:
        expr, expr_error = None, str(exc)
    function_label = "Suggested context function"
else:
    custom_function = st.text_input("Enter f(x), e.g. x^2 - 4*x + 3, exp(-x), sin(x), or 1/(x-2)", "x**2 - 4*x + 3")
    try:
        expr = safe_expression(custom_function)
        expr_error = None
    except ValueError as exc:
        expr, expr_error = None, str(exc)
    function_label = "Custom function"
cc1,cc2,cc3,cc4 = st.columns(4)
dep_name = cc1.text_input("Dependent variable", default_dep)
dep_unit = cc2.text_input("Dependent units", default_dep_unit)
ind_name = cc3.text_input("Independent variable", default_ind)
ind_unit = cc4.text_input("Independent units", default_ind_unit)
context_description = st.text_area("Context description", context_note)
if expr_error:
    st.error(expr_error)
elif expr is not None:
    st.latex("f(x) = " + sp.latex(expr))
    d1_expr, d2_expr = sp.diff(expr, x), sp.diff(expr, x, 2)
    st.latex(r"f'(x) = " + sp.latex(d1_expr))
    st.latex(r"f''(x) = " + sp.latex(d2_expr))
    if function_source == "Curve of best fit":
        domain_default_low, domain_default_high = float(model["min_year"]), float(model["max_year"])
    else:
        domain_default_low, domain_default_high = 0.0, 10.0
    dc1,dc2 = st.columns(2)
    interval_low = dc1.number_input("Analysis interval start", value=float(domain_default_low), key="analysis_low")
    interval_high = dc2.number_input("Analysis interval end", value=float(domain_default_high), key="analysis_high")
    if interval_high <= interval_low:
        st.error("Analysis interval end must be greater than start.")
    else:
        point = st.slider("Current input x", min_value=float(interval_low), max_value=float(interval_high),
                          value=float((interval_low+interval_high)/2), key="current_x")
        ac1, ac2 = st.columns(2)
        secant_a = ac1.slider("Average-rate interval start", min_value=float(interval_low), max_value=float(interval_high),
                              value=float(interval_low), key="secant_a")
        secant_b = ac2.slider("Average-rate interval end", min_value=float(interval_low), max_value=float(interval_high),
                              value=float(interval_high), key="secant_b")
        fval = float(numeric_values(expr, np.array([point]))[0])
        dval = float(numeric_values(d1_expr, np.array([point]))[0])
        ddval = float(numeric_values(d2_expr, np.array([point]))[0])
        if secant_a == secant_b:
            avg_rate = float("nan")
        else:
            fa = float(numeric_values(expr, np.array([secant_a]))[0])
            fb = float(numeric_values(expr, np.array([secant_b]))[0])
            avg_rate = (fb-fa)/(secant_b-secant_a)
        m1,m2,m3 = st.columns(3)
        m1.metric(f"{dep_name} f(x)", f"{fmt_num(fval)} {dep_unit}")
        m2.metric("Instantaneous rate f′(x)", f"{fmt_num(dval)} {dep_unit}/{ind_unit}")
        m3.metric("Second derivative f″(x)", f"{fmt_num(ddval)} {dep_unit}/{ind_unit}²")
        if secant_a == secant_b:
            st.warning("Choose different interval endpoints to calculate average rate of change.")
        else:
            st.write(f"**Average rate of change** from {secant_a:g} to {secant_b:g}: ({fmt_num(fb)} − {fmt_num(fa)}) / ({secant_b:g} − {secant_a:g}) = **{fmt_num(avg_rate)} {dep_unit}/{ind_unit}**.")
        st.write(f"**Context interpretation:** {context_description} At x = {point:g}, the function gives {fmt_num(fval)} {dep_unit}. Its first derivative is the instantaneous rate of change ({fmt_num(dval)} {dep_unit}/{ind_unit}); its second derivative ({fmt_num(ddval)} {dep_unit}/{ind_unit}²) describes how that rate changes.")
        # Graph each function with finite masks; never connect across invalid points.
        graph_x = np.linspace(float(interval_low), float(interval_high), 700)
        try:
            graph_y = numeric_values(expr, graph_x)
            graph_d1 = numeric_values(d1_expr, graph_x)
            graph_d2 = numeric_values(d2_expr, graph_x)
            show_f = st.checkbox("Show f(x)", True)
            show_d1 = st.checkbox("Show f′(x)", course == "Calculus")
            show_d2 = st.checkbox("Show f″(x)", course == "Calculus")
            graph_fig, graph_ax = plt.subplots(figsize=(10,5))
            if show_f:
                graph_ax.plot(graph_x, np.where(np.isfinite(graph_y), graph_y, np.nan), label="f(x)")
            if show_d1:
                graph_ax.plot(graph_x, np.where(np.isfinite(graph_d1), graph_d1, np.nan), label="f′(x)")
            if show_d2:
                graph_ax.plot(graph_x, np.where(np.isfinite(graph_d2), graph_d2, np.nan), label="f″(x)")
            graph_ax.scatter([point], [fval], color="black", zorder=5, label=f"Current point x={point:g}")
            if np.isfinite(dval):
                tangent = fval + dval*(graph_x-point)
                graph_ax.plot(graph_x, tangent, linestyle="--", alpha=.65, label="Tangent at current point")
            if secant_a != secant_b and np.isfinite(avg_rate) and np.isfinite(fa) and np.isfinite(fb):
                secant_x = np.array([secant_a, secant_b])
                secant_y = np.array([fa, fb])
                graph_ax.plot(secant_x, secant_y, linestyle=":", linewidth=2, label="Secant line")
            graph_ax.set_xlabel(f"{ind_name} ({ind_unit})")
            graph_ax.set_ylabel(f"{dep_name} ({dep_unit})")
            graph_ax.set_title(f"{context}: function and rate of change")
            graph_ax.grid(True, alpha=.25); graph_ax.legend(fontsize=8); graph_fig.tight_layout()
            st.pyplot(graph_fig, use_container_width=True)
            if not np.isfinite(fval):
                st.warning("The function is undefined at the selected x. Check for a denominator of zero or a logarithm/square-root domain restriction.")
            # Domain and discontinuity checks on a numeric grid.
            finite_mask = np.isfinite(graph_y)
            if finite_mask.any() and not finite_mask.all():
                st.warning("Some graph values are undefined or non-finite on the selected interval. Gaps are left unconnected.")
            try:
                denom = sp.denom(sp.together(expr))
                restrictions = []
                if denom != 1:
                    restrictions.append(f"Denominator must not equal zero: {sp.latex(denom)} ≠ 0.")
                for logterm in expr.atoms(sp.log):
                    restrictions.append(f"Logarithm argument must be positive: {sp.latex(logterm.args[0])} > 0.")
                for sq in expr.atoms(sp.Pow):
                    if sq.exp.is_Rational and sq.exp.q % 2 == 0:
                        restrictions.append(f"Even-root expression requires a nonnegative radicand: {sp.latex(sq.base)} ≥ 0.")
                st.write("**Domain restrictions:** " + (" ".join(dict.fromkeys(restrictions)) if restrictions else "No obvious denominator, logarithm, or even-root restriction detected."))
            except Exception:
                st.write("**Domain restrictions:** could not symbolically classify all restrictions; inspect graph gaps.")
            if course in ("Precalculus", "Calculus"):
                st.write("**Degree/coefficient effects:** coefficients set vertical scale and direction; degree influences possible turning behavior and end behavior. A numerical graph alone does not prove a real-world cause.")
                roots1 = approximate_real_roots(d1_expr, float(interval_low), float(interval_high))
                roots2 = approximate_real_roots(d2_expr, float(interval_low), float(interval_high))
                st.write("**Changing average rates:** compare secant slopes over different intervals to see whether the function's rate is increasing or decreasing.")
            if course == "Calculus":
                analysis = derivative_analysis(expr, float(interval_low), float(interval_high))
                st.write("**Critical points (numerical candidates; derivative sign test applied):**")
                if analysis["critical"]:
                    for c,yv,kind in analysis["critical"]:
                        st.write(f"- x ≈ {c:.6g}, f(x) ≈ {fmt_num(yv)} — {kind}.")
                else:
                    st.write("- No derivative-zero candidates detected in this interval.")
                st.write("**Increasing/decreasing:** use the sign of f′(x) between critical points; this numerical scan may miss even-multiplicity roots or very narrow features.")
                if analysis["critical"]:
                    bounds = [float(interval_low)] + [z[0] for z in analysis["critical"]] + [float(interval_high)]
                    trend_parts = []
                    for aa,bb in zip(bounds[:-1],bounds[1:]):
                        mid=(aa+bb)/2
                        slope=float(numeric_values(d1_expr,np.array([mid]))[0])
                        trend_parts.append(f"{aa:.4g}–{bb:.4g}: {'increasing' if slope>0 else 'decreasing' if slope<0 else 'approximately flat'}")
                    st.write("; ".join(trend_parts))
                st.write("**Absolute extrema on the selected closed interval (numerical candidates plus endpoints):**")
                if analysis["absolute_min"]:
                    st.write(f"- Minimum candidate: x ≈ {analysis['absolute_min'][0]:.6g}, f(x) ≈ {fmt_num(analysis['absolute_min'][1])}")
                if analysis["absolute_max"]:
                    st.write(f"- Maximum candidate: x ≈ {analysis['absolute_max'][0]:.6g}, f(x) ≈ {fmt_num(analysis['absolute_max'][1])}")
                st.write("**Inflection points (only candidates with a sampled concavity change):**")
                if analysis["inflections"]:
                    for c,yv in analysis["inflections"]:
                        st.write(f"- x ≈ {c:.6g}, f(x) ≈ {fmt_num(yv)}")
                else:
                    st.write("- No concavity-changing inflection point detected numerically.")
                st.write(f"**Concavity / range scan:** over finite sampled graph values, f(x) ranges approximately from {fmt_num(analysis['ymin'])} to {fmt_num(analysis['ymax'])}. Concavity is assessed from the sign of f″(x); this is numerical analysis, not a symbolic proof.")
                st.caption("Numerical approximations are marked with ≈. Symbolic derivative expressions are generated exactly by SymPy for the parsed expression; numerical root-finding can miss roots and does not prove all classifications.")
        except Exception as exc:
            st.error(f"Could not evaluate or graph this function on the selected interval: {exc}")

st.subheader("7. Visual simulation")
visual_options = {
    "Walking person with prosthetic leg": "A simplified gait-cycle position indicator; it is a conceptual visualization, not a medical device model.",
    "Moving object / projectile": "A moving marker follows the selected function. Vertical scaling is based on the visible function range.",
    "Tank / drug concentration": "A fill indicator follows the function value after display-only scaling; it does not represent dosing advice.",
    "Population / cell colony": "A population indicator follows the function value with display-only scaling.",
    "Thermometer / weather": "A thermometer marker follows the function value with display-only scaling.",
    "Business / economic indicator": "A bar indicator follows the function value with display-only scaling.",
}
visual_kind = st.selectbox("Choose contextual visual", list(visual_options))
st.caption(visual_options[visual_kind])
if expr is not None and 'point' in locals() and 'interval_low' in locals() and 'interval_high' in locals() and interval_high > interval_low:
    sample_vals = numeric_values(expr, np.linspace(float(interval_low), float(interval_high), 300))
    finite_vals = sample_vals[np.isfinite(sample_vals)]
    if len(finite_vals):
        ymin_sim, ymax_sim = float(np.min(finite_vals)), float(np.max(finite_vals))
        if math.isclose(ymin_sim, ymax_sim):
            scaled = .5
        else:
            scaled = min(1.0, max(0.0, (fval-ymin_sim)/(ymax_sim-ymin_sim))) if np.isfinite(fval) else 0.0
        sim1,sim2 = st.columns([1,2])
        with sim1:
            st.metric("Current simulation value", f"{fmt_num(fval)} {dep_unit}")
            st.progress(float(scaled), text=f"Display-only scale: {scaled*100:.1f}% of graph range")
            st.caption(f"Scale maps the selected interval's finite range ({fmt_num(ymin_sim)} to {fmt_num(ymax_sim)}) to 0–100%; it does not change the model.")
        with sim2:
            sim_fig, sim_ax = plt.subplots(figsize=(6,2.5))
            sim_ax.barh([visual_kind], [scaled], color="#4c78a8")
            sim_ax.set_xlim(0,1); sim_ax.set_xlabel("Scaled display position (0–1)")
            sim_ax.grid(axis="x", alpha=.25); sim_fig.tight_layout()
            st.pyplot(sim_fig, use_container_width=True)
play_col1, play_col2, play_col3 = st.columns(3)
if "play_x" not in st.session_state:
    st.session_state.play_x = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
with play_col1:
    if st.button("Play playback" if not st.session_state.is_playing else "Pause playback"):
        st.session_state.is_playing = not st.session_state.is_playing
with play_col2:
    if st.button("Reset simulation"):
        st.session_state.play_x = 0
        st.session_state.is_playing = False
with play_col3:
    speed = st.selectbox("Playback speed", ["Slow", "Medium", "Fast"], index=1)
refresh_ms = {"Slow": 1200, "Medium": 600, "Fast": 250}[speed]
if st.session_state.is_playing:
    st.session_state.play_x = (st.session_state.play_x + 1) % 101
    st_autorefresh(interval=refresh_ms, key="simulation_autorefresh")
st.progress(st.session_state.play_x / 100, text=f"Playback step {st.session_state.play_x}/100")
st.write(f"Playback is {'running' if st.session_state.is_playing else 'paused'}. The selected function's current point and tangent remain controlled by the current-input slider; playback progress is a synchronized visual-simulation progress indicator.")

st.subheader("8. Export current report")
edited_status = "User-modified scenario" if st.session_state.edited else "Source-based working data"
meta = {
    "Research question": research_question, "Country": country_label, "Indicator": indicator_name,
    "Indicator code": indicator_code, "Units": unit, "Definition": definition, "Retrieval date": retrieval_date,
    "Requested years": f"{start_year}–{end_year}", "Actual coverage": f"{model['min_year']}–{model['max_year']}",
    "Data status": edited_status, "Course pathway": course, "Context": context,
}
export_results = st.session_state.batch_result.copy() if st.session_state.get("batch_result") is not None else pd.DataFrame(columns=["year","estimate","units","range_status","observed_year"])
if st.session_state.single_result:
    sr = st.session_state.single_result
    single_export = pd.DataFrame([{"year":sr["year"],"estimate":sr["estimate"],"units":unit,"range_status":sr["classification"],"observed_year":sr["is_observed"]}])
    export_results = pd.concat([single_export, export_results], ignore_index=True).drop_duplicates("year", keep="last")
csv_bytes = working_df.to_csv(index=False).encode("utf-8")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    pd.DataFrame([{"field":k,"value":v} for k,v in meta.items()] + [
        {"field":"Polynomial equation","value":equation_text(model)}, {"field":"In-sample R-squared","value":model["r2"]},
        {"field":"In-sample MAE","value":model["mae"]}, {"field":"Observation count","value":model["n"]},
    ]).to_excel(writer, sheet_name="Summary", index=False)
    working_df.to_excel(writer, sheet_name="Working Data", index=False)
    export_results.to_excel(writer, sheet_name="Calculated Years", index=False)
    residual_df.to_excel(writer, sheet_name="Residuals", index=False)
    for sheet in ["Summary","Working Data","Calculated Years","Residuals"]:
        writer.sheets[sheet].freeze_panes(1,0)
        writer.sheets[sheet].set_column(0, 8, 22)
pdf_bytes = build_pdf(meta, fig, working_df, export_results, model)
e1,e2,e3 = st.columns(3)
e1.download_button("Download working data CSV", csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
e2.download_button("Download report Excel (.xlsx)", xlsx_buf.getvalue(), file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
e3.download_button("Download report PDF", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"{country_label} · {indicator_name} ({unit}) · {model['min_year']}–{model['max_year']} · n={model['n']}")
    st.write(equation_text(model))
    st.write(f"In-sample R²: {fmt_num(model['r2'])}; in-sample MAE: {fmt_num(model['mae'])} {unit}.")
    st.write("Interpretation must distinguish observed values from fitted values. Polynomial extrapolation is uncertain and does not establish causation.")

st.divider()
st.subheader("How the app works")
st.markdown("""
- **Data flow:** the app requests indicator observations from the official World Bank API, caches responses for one hour, keeps the source response separate, validates the editable working table, fits a centered-time polynomial, then uses the same fitted coefficients for single-year and batch estimates.
- **Mathematics:** centering time uses `t = year − base year` to reduce numerical scaling problems. Degree `d` requires at least `d + 1` distinct years. R² and MAE are in-sample fit metrics; a chronological holdout MAE is shown separately when there are enough observations.
- **Interpretation:** GDP per capita is not average wealth or household income; crude birth rate is a rate, not a count; net migration is a net measure, not an emigration count. Brazil is a Latin American comparison, but not a Hispanic country.
- **Limitations:** historical coverage differs by country and indicator. Missing API observations remain missing. Polynomial extrapolation is often unstable; no prediction interval is claimed. Numerical calculus root searches can miss roots and should be checked against algebraic reasoning.
- **Context models:** suggested functions are teaching examples, not validated real-world forecasts. Context and units should be customized to match a defensible interpretation.
""")
