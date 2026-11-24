# Predict New York City Airbnb Prices

Complete implementation of **AIML Minor Project 2**, using standard multiple
linear regression on the real `AB_NYC_2019.csv` dataset. The notebook follows all
seven assignment steps and includes executed tables, charts and conclusions.

## Measured results

| Evaluation population | Test listings | R-squared | MAE (USD/night) | RMSE (USD/night) |
|---|---:|---:|---:|---:|
| Typical market after training-derived trimming | 9,164 | 0.4654 | $38.60 | $53.06 |
| Complete positive-price test set | 9,777 | 0.0798 | $63.16 | $191.94 |
| Mean-price baseline on typical-market test rows | 9,164 | -0.000007 | $58.21 | $72.57 |

The model explains about **46.5%** of price variation within the typical-market
test subset and reduces MAE by **33.7%** relative to the mean-price baseline.
Training-derived cutoffs are **$355.70/night** and **30 minimum nights**. On that
subset, inexpensive listings (at most $66) have MAE **$25.86**, versus **$63.69**
for expensive listings (above $160). Relative error is higher for inexpensive
listings, so the conclusion depends on measuring error in dollars. The saved
metrics and one-page report explain these distinctions.

## Open the deliverables

| File | Purpose |
|---|---|
| `NYC_Airbnb_Price_Prediction.ipynb` | Executed, self-contained Jupyter / Colab notebook |
| `NYC_Airbnb_Price_Prediction.html` | Notebook preview with code, charts and results; open in a browser |
| `output/pdf/NYC_Airbnb_Summary_Report.pdf` | Required one-page business summary |
| `data/AB_NYC_2019.csv` | Unmodified dataset, included for offline reruns |
| `data/SOURCE.md` | Original source, mirror URL and verified SHA-256 |
| `output/metrics.json` | Exact measured scores, decisions, counts and package versions |
| `output/figures/` | Six exportable PNG charts |
| `output/test_predictions.csv` | Typical-market held-out predictions and residuals |
| `output/full_test_predictions.csv` | Complete held-out predictions, including outliers |
| `output/airbnb_linear_regression.joblib` | Fitted preprocessing and linear regression model |

The model artifact is generated locally by this project. Use the pinned package
versions when loading it; the notebook can always retrain it from the CSV.

## Rerun on this computer

A Python 3.12 virtual environment is already prepared in `.venv`. From PowerShell
in this project directory:

```powershell
.\.venv\Scripts\python.exe run_project.py
.\.venv\Scripts\python.exe -m pytest -q
```

The runner executes all notebook cells, embeds their outputs, regenerates the HTML
preview and builds the one-page PDF from `output/metrics.json`. It uses a local
Jupyter kernel and does not register a global kernel. The bundled CSV means no
network connection is required to rerun the analysis.

## Set up on another computer

Use **Python 3.12**. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run_project.py
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_project.py
.venv/bin/python -m pytest -q
```

`requirements.txt` records direct dependencies; `requirements-lock.txt` pins the
full environment used for the delivered run. Install packages once; reruns use
the local data and installed libraries. To edit and run interactively, open the
notebook in VS Code or Jupyter and select this Python environment.

For **Google Colab**, upload only the `.ipynb` and choose **Runtime > Run all**.
The analysis uses the standard Colab scientific Python packages. If the CSV is
absent, it downloads the public mirror and verifies the exact file checksum.
Colab package versions may differ slightly from the pinned local run. Download
the generated `output` folder before ending the Colab session. The supplied PDF
is already available; rebuilding it uses `run_project.py` and the report packages.

## Method and interpretation

1. Audit all 48,895 rows and 16 columns, including semantic variable types and
   missing-value counts and percentages.
2. Fill 10,052 missing review rates with zero after verifying that those rows
   have zero reviews. Drop review dates, names and identifiers; remove 11
   zero-price rows. Drop the 221-category neighbourhood field.
3. Reserve 20% of valid-price listings using seed 42. Learn the price and
   minimum-stay 95th percentiles only from the remaining training rows.
4. Apply the frozen cutoffs separately to train and test rows. Report the final
   retained counts. No observations are moved between the two sets.
5. Fit training-only imputation and one-hot encoding with `drop='first'`, then
   ordinary `LinearRegression()` with an intercept. Use seven numerical features,
   borough and room type; no target transformation or prediction clipping.
6. Report R-squared, MAE, MSE and RMSE, a mean-price baseline and complete-test
   scores. Plot actual against predicted prices with an equal-axis diagonal.
7. Plot residuals and compare inexpensive versus expensive listings using
   training-price quartiles, including both dollar and percentage errors.

Steps 3 and 4 define the outlier rule and encoder; learned operations are deferred
until after the split in Step 5 to avoid test-data leakage. Thus, the notebook
keeps the assignment's seven sections while using valid estimation order.

Headline metrics describe a **retrospective typical-market subset**, selected
partly using observed test prices. They do not describe all NYC Airbnb listings,
nor a filter that can be applied to an unknown target at prediction time. The
complete positive-price test results are included to make that scope clear.
An 80/20 split occurs before trimming; independent filtering can slightly change
the final retained proportions.

This is a **2019 listing snapshot**. Advertised nightly prices are not total
trip costs or realized revenue. Random listing validation can share hosts across
sets and does not test new-host or future-year performance. Missing property
size, amenities and quality limit predictive accuracy. Coefficients describe
associations and do not establish causal pricing effects.

## Rebuild the notebook source or submission archive

The notebook is self-contained; it does not import the project scripts. The
builder makes its cells easier to maintain. Rebuilding clears old outputs, so
execute the notebook again afterward:

```powershell
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe run_project.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\package_project.py
```

The archive includes the notebook, HTML preview, PDF, dataset, output evidence,
scripts, tests and dependency files. It excludes virtual environments, caches
and temporary files. A SHA-256 manifest is included for verification.

## Sources

- Original dataset: [Dgomonov, New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).
- Download source: [Alexey Grigorev's public CSV mirror](https://github.com/alexeygrigorev/datasets/blob/master/AB_NYC_2019.csv).
- Implementation references: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
  [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
  [pipelines](https://scikit-learn.org/stable/modules/compose.html), and
  [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
