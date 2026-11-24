"""Build the self-contained, seven-step assignment notebook.

The notebook contains the complete analysis; this file only assembles its cells.
Run from the project root with: python scripts/build_notebook.py
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(r"""
# Predict New York City Airbnb Prices
### AIML Minor Project 2 | Multiple Linear Regression

**Question:** How well can location, room type, stay requirements and listing
activity explain the nightly advertised price of an NYC Airbnb listing?

**Deliverables:** This executed notebook and a separate one-page business report.
All results below are calculated from the real **AB_NYC_2019.csv** dataset.

**Run locally:** Install `requirements.txt`, open this notebook, and select
**Restart Kernel and Run All**. To regenerate every output, run `python run_project.py`.

**Run in Google Colab:** Upload this notebook and choose **Runtime > Run all**.
The notebook downloads the public CSV if it is absent, checks its SHA-256, and
writes outputs under the current directory. Standard Colab scientific Python
packages are sufficient for the analysis. The separate report builder uses the
dependencies in `requirements.txt`.

**Data provenance:** [Original Kaggle dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
by Dgomonov; downloaded from this
[public CSV mirror](https://github.com/alexeygrigorev/datasets/blob/master/AB_NYC_2019.csv).
This is a **2019 snapshot**, not a current NYC price feed. Prices are listed nightly
amounts in USD, rather than verified transaction prices or total trip costs.

**Evaluation design:** `random_state=42`, 80% training and 20% testing before
outlier filtering. All learned preprocessing and 95th-percentile limits use only
training rows. Main scores describe the trimmed, typical-market test subset;
scores on the complete positive-price test set are also disclosed. No target
transformation, model search, or prediction clipping is used.
""")

code(r"""
# Setup: all paths are relative, so this notebook also runs in Colab.
from pathlib import Path
from io import StringIO
import hashlib
import importlib.metadata
import json
import platform
import urllib.request

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

%matplotlib inline

ROOT = Path.cwd()
DATA_PATH = ROOT / 'data' / 'AB_NYC_2019.csv'
OUT = ROOT / 'output'
FIGURES = OUT / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
SEED = 42
QUANTILE = 0.95
EXPECTED_SHA256 = 'e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a'
DATA_URL = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AB_NYC_2019.csv'

if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=90) as response:
        data_bytes = response.read()
    if hashlib.sha256(data_bytes).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Downloaded CSV differs from the verified project dataset.')
    DATA_PATH.write_bytes(data_bytes)

data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
if data_sha256 != EXPECTED_SHA256:
    raise ValueError('CSV checksum mismatch. Restore the verified AB_NYC_2019.csv file.')

sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 160, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white',
})
pd.set_option('display.max_columns', 25)
pd.set_option('display.float_format', lambda value: f'{value:,.3f}')

def save_figure(fig, filename):
    fig.savefig(FIGURES / filename, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)

print(f'Dataset SHA-256 verified. Reproducibility seed: {SEED}')
""")

md(r"""
## Step 1: Exploratory Data Analysis and Data Auditing

### Task 1.1: Load the data and inspect shape, columns and data types
The data audit describes the original file before any rows or columns are removed.
""")

code(r"""
df = pd.read_csv(DATA_PATH)
print(f'Original dataset: {df.shape[0]:,} rows and {df.shape[1]} columns')
print('Columns:', ', '.join(df.columns))
display(df.head())
info_buffer = StringIO()
df.info(buf=info_buffer)
print(info_buffer.getvalue())
display(df.describe().T)
print(f'Exact duplicate rows: {df.duplicated().sum():,}')
print(f'Duplicate listing IDs: {df["id"].duplicated().sum():,}')
""")

md(r"""
### Task 1.2: Classify variables by meaning

Storage type and statistical meaning are different. An integer ID is a nominal
identifier, not a useful continuous measurement. USD prices are treated as
continuous even though this file stores whole dollars. `last_review` is temporal;
listing and host names are free text. No retained feature has a natural ordinal
category order.
""")

code(r"""
semantic_types = {
    'id': 'Nominal identifier (exclude)',
    'name': 'Free text (exclude)',
    'host_id': 'Nominal identifier (exclude)',
    'host_name': 'Free text / nominal identifier (exclude)',
    'neighbourhood_group': 'Categorical - nominal',
    'neighbourhood': 'Categorical - nominal, high cardinality (exclude)',
    'latitude': 'Numerical - continuous',
    'longitude': 'Numerical - continuous',
    'room_type': 'Categorical - nominal',
    'price': 'Numerical - continuous, target (USD/night)',
    'minimum_nights': 'Numerical - discrete count',
    'number_of_reviews': 'Numerical - discrete count',
    'last_review': 'Temporal date string (exclude)',
    'reviews_per_month': 'Numerical - continuous rate',
    'calculated_host_listings_count': 'Numerical - discrete count',
    'availability_365': 'Numerical - discrete count (days)',
}
audit = pd.DataFrame({
    'column': df.columns,
    'dtype': [str(df[col].dtype) for col in df],
    'semantic_type': [semantic_types[col] for col in df],
    'unique_values': [df[col].nunique(dropna=True) for col in df],
    'missing_count': df.isna().sum().to_numpy(),
    'missing_pct': (df.isna().mean() * 100).to_numpy(),
})
display(audit)
audit.to_csv(OUT / 'data_audit.csv', index=False)
""")

md(r"""
### Task 1.3: Identify missing values and their percentages
The table includes every column; the chart focuses on columns with missing values.
""")

code(r"""
missing = audit.loc[audit.missing_count.gt(0), ['column', 'missing_count', 'missing_pct']]
display(missing)
fig, ax = plt.subplots(figsize=(8.5, 3.5), layout='constrained')
ax.barh(missing.column, missing.missing_pct, color='#176b87')
for y, row in enumerate(missing.itertuples()):
    ax.text(row.missing_pct + 0.25, y,
            f'{row.missing_count:,} ({row.missing_pct:.2f}%)', va='center', fontsize=9)
ax.set(xlabel='Missing values (%)', xlim=(0, 28), title='Missingness in the original dataset')
save_figure(fig, '01_missing_values.png')
""")

md(r"""
## Step 2: Data Cleaning and Handling Missing Values

### Task 2.1: Decide what to do with `last_review`
Drop the date string. A defensible alternative would calculate recency relative
to the dataset's extraction date and add a no-review indicator. The assignment
asks for a simple linear model, and the date is missing for many listings, so
dropping it is transparent and avoids an arbitrary date imputation.

### Task 2.2: Investigate missing review rates
Check whether missing `reviews_per_month` occurs on listings with zero reviews.
Only those rows receive **0**; unexpected missing rates remain missing and would
receive a training-set median through the fitted pipeline.

### Task 2.3: Remove irrelevant identifiers
Drop `id`, `host_id`, `name` and `host_name`. Keep the original CSV row index only
as an audit key, never as a model feature. Drop any exact duplicates and remove
nonpositive target prices because a zero nightly price is not a meaningful rate.
""")

code(r"""
missing_rate = df['reviews_per_month'].isna()
review_check = pd.crosstab(
    missing_rate.rename('review_rate_missing'),
    df['number_of_reviews'].eq(0).rename('has_zero_reviews'),
)
display(review_check)
logical_zero = missing_rate & df['number_of_reviews'].eq(0)
unexpected_missing_rates = int((missing_rate & ~df['number_of_reviews'].eq(0)).sum())
print(f'Missing review rates explained by zero reviews: {logical_zero.sum():,}')
print(f'Unexpected missing review rates: {unexpected_missing_rates:,}')

clean = df.drop_duplicates().copy()
clean.loc[clean['reviews_per_month'].isna() & clean['number_of_reviews'].eq(0),
          'reviews_per_month'] = 0.0
invalid_price_count = int((clean['price'].isna() | clean['price'].le(0)).sum())
clean = clean.loc[clean['price'].notna() & clean['price'].gt(0)].copy()
excluded_columns = ['id', 'host_id', 'name', 'host_name', 'last_review']
clean = clean.drop(columns=excluded_columns)
print(f'Removed {invalid_price_count} invalid target prices; {len(clean):,} valid rows remain.')
display(clean.isna().sum().rename('remaining_missing_values').to_frame())
assert clean.index.is_unique
assert clean.price.gt(0).all()
""")

md(r"""
## Step 3: Outlier Identification and Treatment

### Task 3.1: Inspect `price` and `minimum_nights`
The full-range boxplots and histograms expose extreme values without silently
hiding them. Histogram counts use a logarithmic y-axis to make the tail visible.

### Task 3.2: Define the 95th-percentile trimming rule
Retain rows whose `price` **and** `minimum_nights` are at or below their respective
training-set 95th percentiles. This trims records instead of capping their values.
A row above both limits is counted once in the total removal count.

**Order of operations:** Define the rule here; learn and apply the limits in
Step 5 after reserving the 20% test set. Computing limits from all data would let
test-set information influence training. Neither test targets nor test features
are used to choose the limits. Filtering test targets defines a restricted
evaluation population, so the complete-test scores are also reported in Step 6.
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
for column, label, top, bottom in [
    ('price', 'Nightly advertised price (USD)', axes[0, 0], axes[1, 0]),
    ('minimum_nights', 'Minimum stay (nights)', axes[0, 1], axes[1, 1]),
]:
    top.boxplot(clean[column], orientation='horizontal', patch_artist=True,
                boxprops={'facecolor': '#97cadb'},
                flierprops={'markersize': 2, 'alpha': 0.2})
    top.set(xlabel=label, yticks=[], title=f'{column}: full-range boxplot')
    bottom.hist(clean[column], bins=70, color='#176b87', edgecolor='white')
    bottom.set(xlabel=label, ylabel='Listings (log scale)', yscale='log',
               title=f'{column}: full-range histogram')
save_figure(fig, '02_outlier_distributions.png')

def learn_market_limits(training_rows):
    '''Learn outlier thresholds from training data only.'''
    return training_rows[['price', 'minimum_nights']].quantile(QUANTILE)

def typical_market_mask(rows, limits):
    '''Apply frozen thresholds; do not recompute them on evaluation data.'''
    return (rows['price'].gt(0)
            & rows['price'].le(limits['price'])
            & rows['minimum_nights'].le(limits['minimum_nights']))

print('Defined: retain price <= training P95 AND minimum_nights <= training P95.')
""")

md(r"""
## Step 4: Feature Selection and Categorical Encoding

### Task 4.1: Address high cardinality
`neighbourhood` has 221 categories, whereas `neighbourhood_group` has 5 boroughs.
One-hot encoding the former would add up to 220 indicators, including sparse
categories whose effects may be unstable. It does not inherently make regression
impossible, but it increases dimensionality, complicates interpretation, and
introduces redundancy because neighbourhoods are nested inside boroughs.
Drop the fine-grained column as required; retain borough and coordinates.

### Task 4.2: One-hot encode borough and room type
Use `OneHotEncoder(drop='first', handle_unknown='ignore')`. Removing one category
per variable prevents perfect dummy collinearity with the intercept. All-zero
indicators represent the reference category. Categories are learned only when
the pipeline is fitted on training data in Step 6; unseen categories are handled
without inventing a numeric ordering. Such an unseen category is encoded like
the reference, so its estimate should be treated cautiously.

Seven numerical predictors are retained. Numerical median imputation and
categorical mode imputation are fallback safeguards learned on training data;
the verified file has no remaining missing predictor values after logical zeros.
Ordinary least squares does not require feature scaling to obtain fitted prices.
""")

code(r"""
print('Unique neighbourhoods:', clean['neighbourhood'].nunique())
print('Unique boroughs:', clean['neighbourhood_group'].nunique())
clean = clean.drop(columns=['neighbourhood'])

NUMERIC_FEATURES = [
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
]
CATEGORICAL_FEATURES = ['neighbourhood_group', 'room_type']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

preprocessor = ColumnTransformer([
    ('numeric', SimpleImputer(strategy='median'), NUMERIC_FEATURES),
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore',
                                  sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
], remainder='drop', verbose_feature_names_out=False)

print('Predictors:', FEATURES)
print('Encoder configured; no preprocessing has been fitted yet.')
""")

md(r"""
## Step 5: Feature-Target Separation and Validation Split

### Tasks 5.1 and 5.2: Separate X/y and reserve the test set
The random split uses 80% of valid-price rows for training and 20% for testing,
with seed 42. Freeze the split first, then learn 95th-percentile cutoffs on the
training part and filter each part with those same cutoffs. Trimming can slightly
change the final train/test percentage; both counts are printed. No rows are
reassigned after trimming.

This is a listing-level random split. Listings from the same host can appear on
both sides; scores therefore do not establish performance on entirely new hosts
or future years. Grouped or temporal validation would answer those questions.
""")

code(r"""
X = clean[FEATURES].copy()
y = clean['price'].copy()
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.20, random_state=SEED,
)
train_rows = X_train_all.assign(price=y_train_all)
test_rows = X_test_all.assign(price=y_test_all)
limits = learn_market_limits(train_rows)
train_mask = typical_market_mask(train_rows, limits)
test_mask = typical_market_mask(test_rows, limits)

X_train = X_train_all.loc[train_mask].copy()
y_train = y_train_all.loc[train_mask].copy()
X_test = X_test_all.loc[test_mask].copy()
y_test = y_test_all.loc[test_mask].copy()

split_audit = pd.DataFrame([
    {'split': 'train', 'before_trimming': len(X_train_all),
     'removed': int((~train_mask).sum()), 'after_trimming': len(X_train)},
    {'split': 'test', 'before_trimming': len(X_test_all),
     'removed': int((~test_mask).sum()), 'after_trimming': len(X_test)},
])
print('Training-derived 95th-percentile limits:')
display(limits.rename('upper_limit').to_frame())
display(split_audit)
print(f'Training share after trimming: {len(X_train)/(len(X_train)+len(X_test)):.2%}')
assert set(X_train_all.index).isdisjoint(X_test_all.index)
assert len(X_train_all) + len(X_test_all) == len(clean)
assert 'price' not in FEATURES
assert list(X_train.columns) == FEATURES
split_audit.to_csv(OUT / 'split_audit.csv', index=False)

split_membership = pd.DataFrame({'source_row': clean.index})
split_membership['split'] = np.where(split_membership.source_row.isin(X_train_all.index),
                                    'train', 'test')
split_membership['retained'] = split_membership.source_row.isin(
    X_train.index.union(X_test.index))
split_membership.to_csv(OUT / 'split_membership.csv', index=False)
pd.concat([X_train.assign(price=y_train, split='train'),
           X_test.assign(price=y_test, split='test')]).sort_index().to_csv(
    OUT / 'cleaned_model_data.csv', index_label='source_row')
""")

code(r"""
# Describe the retained training market without inspecting test outcomes.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
sns.boxplot(data=X_train.assign(price=y_train), x='room_type', y='price',
            color='#97cadb', showfliers=False, ax=axes[0])
axes[0].set(xlabel='', ylabel='Nightly price (USD)', title='Training prices by room type')
sns.boxplot(data=X_train.assign(price=y_train), x='neighbourhood_group', y='price',
            order=['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
            color='#97cadb', showfliers=False, ax=axes[1])
axes[1].set(xlabel='', ylabel='Nightly price (USD)', title='Training prices by borough')
axes[1].tick_params(axis='x', rotation=20)
save_figure(fig, '03_typical_market_prices.png')
""")

md(r"""
## Step 6: Model Training and Evaluation

### Task 6.1: Fit standard multiple linear regression
The pipeline fits imputers, the one-hot encoder, and `LinearRegression()` using
only retained training rows. The fitted price is

$$\widehat{price}=\beta_0+\sum_{j=1}^{p}\beta_j x_j.$$

### Tasks 6.2 and 6.3: Predict and calculate evaluation metrics
* **R-squared:** Fraction of test price variation explained relative to a
  constant equal to the test-set mean. It is not a percentage of correct predictions.
* **MAE (USD):** Average absolute nightly price error; easier to explain to a business user.
* **MSE (USD squared):** Mean squared error; large mistakes receive extra weight.
* **RMSE (USD):** Square root of MSE; expresses that squared-error penalty in dollars.

A mean-price baseline is fitted on the same retained training targets. The
complete-test evaluation includes luxury and long-stay listings; this exposes
how restricting the target population affects the headline result. Test prices
remain their original USD values, and predictions are neither capped nor clipped.
""")

code(r"""
model = Pipeline([
    ('preprocess', preprocessor),
    ('regression', LinearRegression()),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_pred_full = model.predict(X_test_all)

baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {'n': int(len(actual)), 'r2': float(r2_score(actual, predicted)),
            'mae': float(mean_absolute_error(actual, predicted)),
            'mse': float(mse), 'rmse': float(np.sqrt(mse))}

scores = {
    'typical_market_train': regression_metrics(y_train, y_pred_train),
    'typical_market_test': regression_metrics(y_test, y_pred),
    'mean_baseline_typical_test': regression_metrics(y_test, baseline_pred),
    'full_positive_price_test': regression_metrics(y_test_all, y_pred_full),
}
score_table = pd.DataFrame(scores).T.rename_axis('evaluation_scope')
display(score_table)
score_table.to_csv(OUT / 'evaluation_metrics.csv')

encoder = model.named_steps['preprocess'].named_transformers_['categorical'].named_steps['encoder']
reference_categories = {
    column: str(categories[0])
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_)
}
feature_names = model.named_steps['preprocess'].get_feature_names_out()
coefficient_table = pd.DataFrame({
    'feature': feature_names,
    'coefficient_usd_per_unit': model.named_steps['regression'].coef_,
})
print(f'Encoded feature count: {len(feature_names)}')
print('Reference categories:', reference_categories)
print(f'Intercept: {model.named_steps["regression"].intercept_:.3f}')
display(coefficient_table)
coefficient_table.to_csv(OUT / 'linear_regression_coefficients.csv', index=False)
print('Negative test predictions:', int((y_pred < 0).sum()))
""")

md(r"""
**Coefficient interpretation:** A room-type or borough coefficient is an
adjusted dollar difference relative to its printed reference category, holding
other included features constant. A numerical coefficient is the fitted change
for one unit of that predictor; a whole degree of latitude or longitude is much
larger than a typical local move. These associations are not causal effects.
The intercept evaluates coordinates of zero, outside NYC, so it has no useful
standalone business interpretation. Negative prices are possible with unconstrained
ordinary least squares and are counted above instead of silently corrected.
""")

md(r"""
## Step 7: Residual Analysis and Interpretation

### Task 7.1: Actual vs. predicted prices
The dashed 45-degree line represents perfect predictions. Equal axis limits and
equal aspect ratio make the diagonal visually meaningful. Points below the line
are underestimated; points above it are overestimated.
""")

code(r"""
predictions = pd.DataFrame({
    'actual_price': y_test, 'predicted_price': y_pred,
    'borough': X_test['neighbourhood_group'], 'room_type': X_test['room_type'],
})
predictions.index.name = 'source_row'
predictions['residual'] = predictions.actual_price - predictions.predicted_price
predictions['absolute_error'] = predictions.residual.abs()

fig, ax = plt.subplots(figsize=(6.7, 6.2), layout='constrained')
ax.scatter(predictions.actual_price, predictions.predicted_price,
           s=8, alpha=0.17, color='#176b87', edgecolors='none')
lower = min(0, float(predictions.predicted_price.min())) - 5
upper = max(float(y_test.max()), float(predictions.predicted_price.max())) + 5
ax.plot([lower, upper], [lower, upper], '--', color='#d76b36', linewidth=2,
        label='Perfect prediction (45 degrees)')
ax.set(xlim=(lower, upper), ylim=(lower, upper), aspect='equal',
       xlabel='Actual nightly price (USD)', ylabel='Predicted nightly price (USD)',
       title='Actual vs. predicted | typical-market test set')
ax.legend(loc='upper left', fontsize=9)
save_figure(fig, '04_actual_vs_predicted.png')
""")

md(r"""
### Task 7.2: Residual plot and error distribution
Residual means **actual minus predicted**. Positive values mean underpricing;
negative values mean overpricing. A well-behaved linear model would show a fairly
random cloud around zero, with roughly constant spread. A changing spread or
pattern suggests remaining structure or unequal error variance.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
axes[0].scatter(predictions.predicted_price, predictions.residual,
                s=7, alpha=0.16, color='#176b87', edgecolors='none')
axes[0].axhline(0, color='#d76b36', linestyle='--', linewidth=1.5)
axes[0].set(xlabel='Predicted nightly price (USD)', ylabel='Residual (USD)',
            title='Residuals vs. predicted price')
axes[1].hist(predictions.residual, bins=55, color='#176b87', edgecolor='white')
axes[1].axvline(0, color='#d76b36', linestyle='--', linewidth=1.5)
axes[1].set(xlabel='Residual: actual minus predicted (USD)', ylabel='Test listings',
            title='Distribution of prediction errors')
save_figure(fig, '05_residual_analysis.png')
""")

md(r"""
**Observed pattern:** The actual-versus-predicted plot compresses estimates toward
central prices: many high-price listings fall below the perfect-prediction line.
The residual plot has a changing spread and the histogram has a long positive
tail, consistent with some substantial underestimates. The residual boundary is
also affected by the deliberate ceiling on actual test prices. These plots do
not support assuming equal error variance or a symmetric normal error distribution.
The price-band calculation below measures the dollar-error pattern directly.
""")

md(r"""
### Task 7.3: Does the model perform better on cheap or expensive listings?
Define inexpensive and expensive using the **retained training prices' 25th and
75th percentiles**, so the comparison boundaries are fixed before looking at test
errors. Compare absolute dollar errors, signed residuals, and relative errors.
The middle band is also reported. Here MAPE is the mean absolute percentage
error; very low actual prices can make it large.
""")

code(r"""
q25, q75 = y_train.quantile([0.25, 0.75]).to_numpy()
band_order = ['Inexpensive (bottom quartile)', 'Middle half', 'Expensive (top quartile)']
predictions['price_band'] = pd.cut(
    predictions.actual_price, bins=[-np.inf, q25, q75, np.inf],
    labels=band_order, right=True,
)
band_results = []
for label in band_order:
    part = predictions.loc[predictions.price_band.eq(label)]
    metrics = regression_metrics(part.actual_price, part.predicted_price)
    band_results.append({
        'price_band': label, **metrics,
        'mean_actual_price': float(part.actual_price.mean()),
        'mean_residual': float(part.residual.mean()),
        'mape_pct': float((part.absolute_error / part.actual_price).mean() * 100),
    })
band_table = pd.DataFrame(band_results).set_index('price_band')
print(f'Inexpensive <= ${q25:.2f}; expensive > ${q75:.2f}.')
display(band_table[['n', 'mean_actual_price', 'mae', 'rmse', 'mean_residual', 'mape_pct']])
band_table.to_csv(OUT / 'error_by_price_band.csv')

cheap = band_table.loc[band_order[0]]
expensive = band_table.loc[band_order[2]]
better = 'inexpensive' if cheap.mae < expensive.mae else 'expensive'
interpretation = (
    f'The model performs better on **{better} listings in absolute dollar terms**. '
    f'MAE is **${cheap.mae:.2f}** for prices up to ${q25:.0f}, '
    f'compared with **${expensive.mae:.2f}** for prices above ${q75:.0f}. '
    f'The expensive band has a mean actual-minus-predicted residual of '
    f'**${expensive.mean_residual:.2f}**. '
    'A positive residual indicates underestimation. '
    'Ordinary least squares pulls fitted values toward the central price pattern; '
    'larger prices can have larger dollar errors. Unobserved size, amenities, '
    'quality and nonlinear location effects are plausible additional explanations, '
    'rather than causes established by this dataset. '
    f'Relative errors tell a different part of the story: MAPE is '
    f'{cheap.mape_pct:.1f}% for inexpensive listings and '
    f'{expensive.mape_pct:.1f}% for expensive listings. '
    'Therefore, lower dollar error does not automatically mean better percentage accuracy.'
)
display(Markdown(interpretation))

fig, ax = plt.subplots(figsize=(8.5, 3.7), layout='constrained')
bars = ax.bar(['Inexpensive', 'Middle', 'Expensive'], band_table.mae,
              color=['#176b87', '#71a9b9', '#d76b36'], width=0.55)
for bar, value in zip(bars, band_table.mae):
    ax.text(bar.get_x() + bar.get_width()/2, value + 1.2,
            f'${value:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set(ylabel='Mean absolute error (USD/night)',
       title='Where the model makes larger dollar errors',
       ylim=(0, float(band_table.mae.max()) * 1.23))
save_figure(fig, '06_error_by_price_band.png')
""")

md(r"""
### Business conclusion and limitations
The next cell summarizes the computed results in plain English and saves the
evidence used by the separate one-page report. Scores are empirical results of
this run, not guaranteed future accuracy.

The trimmed test set is selected partly by its observed target price; this is a
conditional retrospective analysis, not an online method for deciding whether
an unknown-price listing is in scope. Dropping expensive listings deliberately
changes the problem. Random holdout evaluation also does not establish temporal
generalization. Missing size, bedroom, amenity and quality variables limit what
the model can explain. Review and availability variables are snapshot features
and may themselves reflect past prices and demand. Coefficients describe
associations, not causal drivers. Residual plots and band errors are diagnostic;
they do not by themselves prove normality or constant variance.
""")

code(r"""
main = scores['typical_market_test']
base = scores['mean_baseline_typical_test']
full = scores['full_positive_price_test']
mae_improvement = (1 - main['mae'] / base['mae']) * 100
business_summary = (
    f'On {main["n"]:,} held-out typical-market listings, R-squared is '
    f'**{main["r2"]:.4f}**: the model explains about **{main["r2"]*100:.1f}%** '
    'of variation in listed nightly prices. This is not classification accuracy. '
    f'MAE is **${main["mae"]:.2f} per night**, meaning that the average prediction '
    f'misses the listed price by that amount in either direction. RMSE is '
    f'**${main["rmse"]:.2f}**, which puts extra weight on larger misses. '
    f'The mean-price baseline has MAE **${base["mae"]:.2f}**, so the model reduces '
    f'average absolute error by **{mae_improvement:.1f}%** against that baseline. '
    f'These results concern prices at or below **${limits["price"]:.2f}** and '
    f'minimum stays at or below **{limits["minimum_nights"]:.0f} nights**. '
    f'On all {full["n"]:,} held-out positive-price listings, including excluded '
    f'luxury and long-stay rows, MAE is **${full["mae"]:.2f}** and R-squared is '
    f'**{full["r2"]:.4f}**. Use the estimate as an initial benchmark, with human '
    'review and current comparable listings before choosing a real asking price.'
)
display(Markdown(business_summary))

versions = {name: importlib.metadata.version(name) for name in [
    'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib',
]}
results = {
    'project': 'Predict New York City Airbnb Prices',
    'dataset': {
        'filename': DATA_PATH.name, 'rows': int(len(df)), 'columns': int(df.shape[1]),
        'sha256': data_sha256, 'download_url': DATA_URL,
        'original_source': 'https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data',
        'snapshot_year': 2019,
    },
    'random_state': SEED, 'test_size': 0.20, 'outlier_quantile': QUANTILE,
    'outlier_limits': {k: float(v) for k, v in limits.items()},
    'outlier_limits_fit_scope': 'training rows before trimming only',
    'missing_review_rates_filled_zero': int(logical_zero.sum()),
    'unexpected_missing_review_rates': unexpected_missing_rates,
    'invalid_price_rows_removed': invalid_price_count,
    'exact_duplicate_rows': int(df.duplicated().sum()),
    'split_counts': split_audit.to_dict(orient='records'),
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'encoded_feature_count': int(len(feature_names)),
    'reference_categories': reference_categories,
    'metrics': scores,
    'mae_improvement_vs_mean_baseline_pct': float(mae_improvement),
    'price_band_limits': {'inexpensive_at_most': float(q25), 'expensive_above': float(q75)},
    'price_band_results': band_table.reset_index().to_dict(orient='records'),
    'negative_predictions_typical_test': int((y_pred < 0).sum()),
    'business_summary': business_summary.replace('**', ''),
    'price_band_interpretation': interpretation.replace('**', ''),
    'python_version': platform.python_version(), 'package_versions': versions,
}
(OUT / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
predictions.to_csv(OUT / 'test_predictions.csv')
pd.DataFrame({
    'source_row': X_test_all.index,
    'actual_price': y_test_all.to_numpy(),
    'predicted_price': y_pred_full,
    'in_typical_market': test_mask.to_numpy(),
}).to_csv(OUT / 'full_test_predictions.csv', index=False)
joblib.dump({'model': model, 'market_limits': limits.to_dict(),
             'features': FEATURES, 'dataset_sha256': data_sha256,
             'package_versions': versions}, OUT / 'airbnb_linear_regression.joblib')
print('Saved figures, audit tables, split membership, predictions, metrics and fitted model to:', OUT)
""")

md(r"""
## Requirement coverage

| Assignment requirement | Evidence in this notebook |
|---|---|
| 1.1 Shape, columns, dtypes | Original-data inspection and `df.info()` |
| 1.2 Numerical / categorical classification | Complete semantic data dictionary |
| 1.3 Missing counts and percentages | Audit table and missingness chart |
| 2.1 Review date decision | Explained drop of `last_review` |
| 2.2 Logical review-rate imputation | Crosstab and zero fill only for zero-review rows |
| 2.3 Remove identifiers | Explicit five-column exclusion list |
| 3.1 Outlier visuals | Full-range price and minimum-stay boxplots / histograms |
| 3.2 95th-percentile treatment | Training-derived limits; counts before / after |
| 4.1 High cardinality | Drop neighbourhood; explain dimensionality and sparsity |
| 4.2 One-hot encoding, first category dropped | Fitted pipeline and printed reference categories |
| 5.1 Features and target | Explicit X/y selection |
| 5.2 Reproducible 80/20 split | `train_test_split(test_size=0.20, random_state=42)` |
| 6.1 Standard Linear Regression | `LinearRegression()` in training-only pipeline |
| 6.2 Test predictions | Held-out predictions, exported as CSV |
| 6.3 R-squared, MAE, MSE / RMSE | Metric table plus business interpretation |
| 7.1 Actual vs. predicted with diagonal | Equal-axis scatter and 45-degree reference line |
| 7.2 Residual analysis | Residual scatter and error histogram |
| 7.3 Cheap vs. expensive | Training-defined bands, dollar and percentage errors |
| One-page report | `output/pdf/NYC_Airbnb_Summary_Report.pdf` in the project bundle |

### References
* Dgomonov, [New York City Airbnb Open Data (2019), Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).
* Alexey Grigorev, [public mirror of AB_NYC_2019.csv](https://github.com/alexeygrigorev/datasets/blob/master/AB_NYC_2019.csv).
* Scikit-learn, [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
  [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
  [pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html),
  and [common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
""")


if __name__ == '__main__':
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'colab': {'provenance': []},
    }
    nbf.validate(nb)
    destination = ROOT / 'NYC_Airbnb_Price_Prediction.ipynb'
    nbf.write(nb, destination)
    print(f'Built {destination.name}: {len(cells)} cells')
