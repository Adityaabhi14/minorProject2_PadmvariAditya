"""Bundle the finished project without local environments or temporary files."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]


def main():
    required = [
        '.gitignore', '.gitattributes',
        'README.md', 'requirements.txt', 'requirements-lock.txt', 'run_project.py',
        'NYC_Airbnb_Price_Prediction.ipynb', 'NYC_Airbnb_Price_Prediction.html',
        'data/AB_NYC_2019.csv', 'data/SOURCE.md',
        'scripts/build_notebook.py', 'scripts/create_report.py',
        'scripts/package_project.py', 'tests/test_deliverables.py',
    ]
    files = [ROOT / name for name in required]
    files += sorted(path for path in (ROOT / 'output').rglob('*') if path.is_file())
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required deliverable: {path}')
    if not (ROOT / 'output/pdf/NYC_Airbnb_Summary_Report.pdf').is_file():
        raise FileNotFoundError('Run the project to generate the required report first.')
    manifest = ''.join(
        f'{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n'
        for path in files
    )
    destination = ROOT / 'submission'
    destination.mkdir(exist_ok=True)
    (destination / 'MANIFEST.sha256').write_text(manifest, encoding='utf-8')
    archive = destination / 'NYC_Airbnb_Project.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=7) as package:
        for path in files:
            package.write(path, path.relative_to(ROOT).as_posix())
        package.writestr('MANIFEST.sha256', manifest)
    print(f'Created {archive} ({archive.stat().st_size / 1024**2:.2f} MiB; {len(files)} files)')


if __name__ == '__main__':
    main()
