# Hispanic Heritage Historical Data Explorer

A Streamlit classroom app for retrieving authentic World Bank historical indicators, editing a working copy, fitting centered-time polynomial models, exploring functions and derivatives, comparing countries, and exporting CSV, Excel, and PDF reports.

## Before you begin

Replace the default research question in the app with your team's actual question. Choose a country, indicator, and course pathway that match your assignment. The app uses the official World Bank API and does not generate replacement data when a request fails.

## Run locally

Use Python 3.10 or newer (Python 3.11 is a good default).

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud and create an app from that repository.
4. Set the main file path to `app.py`, then deploy.
5. Review the app logs if dependency installation or API requests fail.

No API key or credentials are required.

## Deploy on Render

Create a **Python Web Service** connected to the GitHub repository.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

## Verification checklist

- [ ] Country and indicator change the World Bank request and labels.
- [ ] Requested period is current year minus 70 through current year minus 1; displayed coverage/count reflect non-null observations actually returned.
- [ ] API timeout/error or empty response shows a message and does not create fake data.
- [ ] Edit a year/value; verify the scenario warning and exports update. Press **Reset to source** to restore source observations.
- [ ] Try a duplicate year, fractional year, nonnumeric value, negative unemployment, or life expectancy above 125; verify validation catches invalid data.
- [ ] Degree d requires at least d+1 distinct years.
- [ ] Equation, base year, in-sample R², MAE, residuals, observation count, mean, and historical range are displayed.
- [ ] Calculate the same year using the single-year and batch controls; estimates should match to floating-point rounding.
- [ ] Try an invalid custom year such as `20xx`; confirm it is explained/ignored.
- [ ] Change sample increment, forward horizon, comparison countries, course pathway, function source, graph visibility checkboxes, analysis interval, current point, and playback speed; verify their outputs change.
- [ ] Download CSV, `.xlsx`, and PDF; confirm Excel has Summary, Working Data, Calculated Years, and Residuals sheets, and the PDF contains metadata and a chart.
- [ ] On deployment, check app logs for package errors, blocked network calls, and API response errors.

## How to defend the mathematics

- The polynomial uses `t = year − base year` rather than raw calendar year to improve numerical conditioning.
- Degree `d` needs at least `d+1` distinct years. Higher degree can follow historical data more closely but can oscillate and extrapolate poorly.
- R² and MAE are explicitly **in-sample**. A chronological holdout score, when available, uses earlier observations for fitting and later observations for evaluation.
- A residual is actual value minus fitted value. MAE is the average absolute residual.
- The derivative is the instantaneous rate of change; the second derivative describes how that rate changes. Units are dependent units per independent unit, and dependent units per independent-unit squared.
- Calculus root and inflection detection is numerical. It uses derivative sign checks for classifications and should be treated as approximate, especially for complicated expressions or discontinuities.
- No prediction interval is claimed. MAE/residuals describe observed fit error, not a guarantee about future values.
