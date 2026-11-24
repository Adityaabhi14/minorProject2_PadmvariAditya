"""Integration checks against the real dataset and saved deliverables.

Run after `python run_project.py` with `python -m pytest -q`.
"""

import hashlib
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = 'e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a'


@pytest.fixture(scope='module')
def results():
    return json.loads((ROOT / 'output' / 'metrics.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def raw():
    return pd.read_csv(ROOT / 'data' / 'AB_NYC_2019.csv')


@pytest.fixture(scope='module')
def bundle():
    return joblib.load(ROOT / 'output' / 'airbnb_linear_regression.joblib')


@pytest.fixture(scope='module')
def predictions():
    return pd.read_csv(ROOT / 'output' / 'test_predictions.csv').set_index('source_row')


def test_original_data_is_real_and_unchanged(raw, results):
    digest = hashlib.sha256((ROOT / 'data' / 'AB_NYC_2019.csv').read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256 == results['dataset']['sha256']
    assert raw.shape == (48895, 16)
    assert raw.price.eq(0).sum() == results['invalid_price_rows_removed'] == 11
    missing_rate = raw.reviews_per_month.isna()
    assert missing_rate.sum() == 10052
    assert raw.loc[missing_rate, 'number_of_reviews'].eq(0).all()


def test_notebook_is_fully_executed_and_covers_seven_steps():
    notebook = nbformat.read(ROOT / 'NYC_Airbnb_Price_Prediction.ipynb', as_version=4)
    nbformat.validate(notebook)
    markdown = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'markdown')
    for step in range(1, 8):
        assert f'## Step {step}:' in markdown
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code_cells)
    assert [c.execution_count for c in code_cells] == list(range(1, len(code_cells) + 1))
    assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']
    chart_count = sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
    assert chart_count >= 6
    assert (ROOT / 'NYC_Airbnb_Price_Prediction.html').stat().st_size > 100_000


def test_split_is_reproducible_disjoint_and_thresholds_use_training_only(raw, results):
    membership = pd.read_csv(ROOT / 'output' / 'split_membership.csv').set_index('source_row')
    valid = raw.loc[raw.price.gt(0)]
    expected_train, expected_test = train_test_split(valid.index, test_size=.2, random_state=42)
    actual_train = membership.index[membership.split.eq('train')]
    actual_test = membership.index[membership.split.eq('test')]
    assert set(actual_train) == set(expected_train)
    assert set(actual_test) == set(expected_test)
    assert set(actual_train).isdisjoint(actual_test)
    assert set(membership.index) == set(valid.index)
    expected_limits = raw.loc[expected_train, ['price', 'minimum_nights']].quantile(.95)
    for feature, threshold in results['outlier_limits'].items():
        assert threshold == pytest.approx(expected_limits[feature])
    expected_retained = (
        raw.loc[membership.index, 'price'].le(expected_limits.price)
        & raw.loc[membership.index, 'minimum_nights'].le(expected_limits.minimum_nights)
    )
    np.testing.assert_array_equal(membership.retained, expected_retained)
    for part in ['train', 'test']:
        counts = next(v for v in results['split_counts'] if v['split'] == part)
        selected = membership.loc[membership.split.eq(part)]
        assert len(selected) == counts['before_trimming']
        assert selected.retained.sum() == counts['after_trimming']


def test_pipeline_uses_training_statistics_and_drop_first(raw, results, bundle):
    model = bundle['model']
    assert isinstance(model.named_steps['regression'], LinearRegression)
    forbidden = {'price', 'id', 'host_id', 'name', 'host_name', 'last_review', 'neighbourhood'}
    assert forbidden.isdisjoint(bundle['features'])
    assert len(bundle['features']) == 9
    assert model.named_steps['regression'].n_features_in_ == 13
    encoder = model.named_steps['preprocess'].named_transformers_['categorical'].named_steps['encoder']
    assert encoder.drop == 'first'
    assert encoder.handle_unknown == 'ignore'
    train = pd.read_csv(ROOT / 'output' / 'cleaned_model_data.csv')
    train = train.loc[train.split.eq('train')]
    medians = train[results['numeric_features']].median().to_numpy()
    imputer = model.named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(imputer.statistics_, medians)
    for col, cats in zip(results['categorical_features'], encoder.categories_):
        assert list(cats) == sorted(train[col].unique())
    example = train[bundle['features']].iloc[[0]].copy()
    example.loc[:, 'neighbourhood_group'] = 'Unseen borough'
    example.loc[:, 'reviews_per_month'] = np.nan
    with pytest.warns(UserWarning, match='unknown categories'):
        assert np.isfinite(model.predict(example)).all()


def test_predictions_and_reported_metrics_match_unmodified_targets(raw, results, bundle, predictions):
    sample = raw.loc[predictions.index, bundle['features']].copy()
    zero_review_missing = sample.reviews_per_month.isna() & sample.number_of_reviews.eq(0)
    sample.loc[zero_review_missing, 'reviews_per_month'] = 0.0
    np.testing.assert_allclose(predictions.actual_price, raw.loc[predictions.index, 'price'])
    np.testing.assert_allclose(predictions.predicted_price, bundle['model'].predict(sample))
    residual = predictions.actual_price.to_numpy() - predictions.predicted_price.to_numpy()
    main = results['metrics']['typical_market_test']
    assert main['n'] == len(predictions)
    assert main['mae'] == pytest.approx(np.abs(residual).mean())
    assert main['mse'] == pytest.approx(np.square(residual).mean())
    assert main['rmse'] == pytest.approx(np.sqrt(np.square(residual).mean()))
    total_variation = np.square(predictions.actual_price - predictions.actual_price.mean()).sum()
    assert main['r2'] == pytest.approx(1 - np.square(residual).sum() / total_variation)
    assert main['rmse'] < results['metrics']['mean_baseline_typical_test']['rmse']
    membership = pd.read_csv(ROOT / 'output' / 'split_membership.csv').set_index('source_row')
    expected = membership.index[membership.split.eq('test') & membership.retained]
    assert set(predictions.index) == set(expected)


def test_full_test_scores_are_disclosed_and_not_mislabeled(raw, results, predictions):
    full = pd.read_csv(ROOT / 'output' / 'full_test_predictions.csv')
    assert len(full) == results['metrics']['full_positive_price_test']['n']
    assert len(full) > len(predictions)
    np.testing.assert_allclose(full.actual_price, raw.loc[full.source_row, 'price'])
    assert set(full.loc[full.in_typical_market, 'source_row']) == set(predictions.index)
    assert results['metrics']['full_positive_price_test']['mae'] == pytest.approx(
        np.abs(full.actual_price - full.predicted_price).mean())


def test_price_band_conclusion_has_numerical_evidence(results, predictions):
    train = pd.read_csv(ROOT / 'output' / 'cleaned_model_data.csv')
    q25, q75 = train.loc[train.split.eq('train'), 'price'].quantile([.25, .75])
    assert results['price_band_limits'] == {'inexpensive_at_most': q25, 'expensive_above': q75}
    band_rows = results['price_band_results']
    assert sum(row['n'] for row in band_rows) == len(predictions)
    cheap = predictions.loc[predictions.actual_price.le(q25)]
    expensive = predictions.loc[predictions.actual_price.gt(q75)]
    assert band_rows[0]['mae'] == pytest.approx(cheap.absolute_error.mean())
    assert band_rows[2]['mae'] == pytest.approx(expensive.absolute_error.mean())
    assert cheap.absolute_error.mean() < expensive.absolute_error.mean()
    assert expensive.residual.mean() > 0


def test_report_is_exactly_one_page_and_uses_computed_results(results):
    reader = PdfReader(ROOT / 'output' / 'pdf' / 'NYC_Airbnb_Summary_Report.pdf')
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    main = results['metrics']['typical_market_test']
    assert f'{main["r2"]:.4f}' in text
    assert f'${main["mae"]:.2f}' in text
    assert '2019' in text
    assert 'training-only' in text
    assert 'unknown future price' in text
    assert f'${results["outlier_limits"]["price"]:.2f}' in text
    assert len(reader.pages[0].get('/Annots', [])) >= 2
    for path in (ROOT / 'output' / 'figures').glob('*.png'):
        with Image.open(path) as img:
            assert img.width >= 900 and img.height >= 400
            img.verify()
