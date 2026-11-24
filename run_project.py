"""Execute the notebook and regenerate its HTML preview and one-page report.

Usage (from the project directory): python run_project.py
No user-level Jupyter kernel registration or system changes are required.
"""

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
TEMP = ROOT / 'tmp'
TEMP.mkdir(exist_ok=True)
for key, subdirectory in [
    ('MPLCONFIGDIR', 'matplotlib'),
    ('IPYTHONDIR', 'ipython'),
    ('JUPYTER_RUNTIME_DIR', 'jupyter'),
]:
    destination = TEMP / subdirectory
    destination.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(destination)
os.environ['PYTHONUTF8'] = '1'

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from scripts.create_report import create_report


def main():
    import json

    notebook_path = ROOT / 'NYC_Airbnb_Price_Prediction.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    # Use this exact Python environment for the kernel, including on Windows.
    spec_root = TEMP / 'kernels'
    spec_directory = spec_root / 'airbnb-project'
    spec_directory.mkdir(parents=True, exist_ok=True)
    (spec_directory / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Airbnb project Python', 'language': 'python',
    }), encoding='utf-8')
    manager = KernelManager(
        kernel_name='airbnb-project',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_root)]),
        connection_file=str(TEMP / 'airbnb-kernel.json'),
    )
    client = NotebookClient(
        notebook, km=manager, timeout=240,
        resources={'metadata': {'path': str(ROOT)}},
        allow_errors=False, record_timing=False,
    )
    code_count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'Executing {code_count} analysis cells using {sys.executable}', flush=True)

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            first_line = cell.source.splitlines()[0][:85]
            print(f'  Cell {cell_index + 1}: {first_line}', flush=True)

    client.on_cell_start = progress
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    print('Saved executed notebook.', flush=True)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(notebook)
    (ROOT / 'NYC_Airbnb_Price_Prediction.html').write_text(html, encoding='utf-8')
    print('Saved HTML preview.', flush=True)
    print(f'Saved one-page report: {create_report(ROOT)}', flush=True)
    results = json.loads((ROOT / 'output' / 'metrics.json').read_text(encoding='utf-8'))
    scores = results['metrics']['typical_market_test']
    print(f'Complete: R2={scores["r2"]:.6f}, MAE=${scores["mae"]:.4f}, '
          f'RMSE=${scores["rmse"]:.4f}, test n={scores["n"]:,}', flush=True)


if __name__ == '__main__':
    main()
