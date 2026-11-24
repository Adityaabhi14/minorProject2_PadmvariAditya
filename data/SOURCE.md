# Dataset provenance

- Dataset: **New York City Airbnb Open Data (2019)**, commonly distributed as `AB_NYC_2019.csv`.
- Original publisher page: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>.
- Downloaded public CSV mirror: <https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AB_NYC_2019.csv>.
- Mirror file page: <https://github.com/alexeygrigorev/datasets/blob/master/AB_NYC_2019.csv>.
- Retrieved for this project: **24 November 2026**. The observations are from **2019**.
- Verified dimensions: **48,895 rows x 16 columns**.
- SHA-256: `e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a`.

The original CSV is preserved without modification. The notebook checks this
checksum before analysis and after an automatic download. If the mirror changes,
the notebook fails explicitly instead of silently analyzing a different file.
Keep the bundled CSV to rerun without a network connection.

Prices are advertised nightly USD rates in a historical listing snapshot. They
are not current price recommendations, realized sales, or total trip costs. No
claim of direct collection from Airbnb is made.
