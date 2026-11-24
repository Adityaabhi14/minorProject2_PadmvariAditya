"""Create the assignment's one-page report from measured notebook outputs."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor('#142d40')
TEAL = colors.HexColor('#176b87')
TEXT = colors.HexColor('#243b4a')
MUTED = colors.HexColor('#556975')
PALE = colors.HexColor('#edf4f7')
ORANGE = colors.HexColor('#b9552c')


def create_report(project_root=ROOT):
    project_root = Path(project_root)
    data = json.loads((project_root / 'output' / 'metrics.json').read_text(encoding='utf-8'))
    main = data['metrics']['typical_market_test']
    base = data['metrics']['mean_baseline_typical_test']
    full = data['metrics']['full_positive_price_test']
    cheap, _, expensive = data['price_band_results']
    limits = data['outlier_limits']
    bands = data['price_band_limits']
    train = data['split_counts'][0]
    test = data['split_counts'][1]
    folder = project_root / 'output' / 'pdf'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'NYC_Airbnb_Summary_Report.pdf'
    page_w, page_h = A4
    c = canvas.Canvas(str(destination), pagesize=A4)
    c.setTitle('Predict New York City Airbnb Prices | One-Page Summary')
    c.setAuthor('AIML Minor Project 2')
    margin = 39
    width = page_w - 2 * margin

    def paragraph(text, x, top, available_width=width, size=10.2, leading=14.4,
                  color=TEXT, font='Helvetica', max_height=100):
        style = ParagraphStyle('body', fontName=font, fontSize=size, leading=leading,
                               textColor=color, alignment=TA_LEFT)
        p = Paragraph(text, style)
        _, height = p.wrap(available_width, max_height)
        if height > max_height:
            raise ValueError(f'Report paragraph exceeds its layout box: {height} > {max_height}')
        p.drawOn(c, x, top - height)
        return top - height

    def section(title, top):
        c.setFillColor(TEAL)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(margin, top, title.upper())
        return top - 12

    # Header and measured performance cards.
    c.setFillColor(NAVY)
    c.rect(0, page_h - 113, page_w, 113, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#a9d4e3'))
    c.setFont('Helvetica-Bold', 9)
    c.drawString(margin, page_h - 29, 'AIML MINOR PROJECT 2  /  BUSINESS SUMMARY')
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 22)
    c.drawString(margin, page_h - 59, 'Predict NYC Airbnb Prices')
    paragraph('Multiple linear regression | 2019 listed nightly prices in USD',
              margin, page_h - 76, size=10, color=colors.HexColor('#d7e7ee'), max_height=26)

    top = page_h - 129
    gap = 10
    card_width = (width - 2 * gap) / 3
    for i, (label, value, detail) in enumerate([
        ('R-SQUARED', f'{main["r2"]:.4f}', f'{main["r2"]*100:.1f}% of price variation explained'),
        ('MEAN ABSOLUTE ERROR', f'${main["mae"]:.2f}', 'Average nightly price miss'),
        ('ROOT MEAN SQUARED ERROR', f'${main["rmse"]:.2f}', 'More weight on larger misses'),
    ]):
        x = margin + i * (card_width + gap)
        c.setFillColor(PALE)
        c.roundRect(x, top - 75, card_width, 75, 7, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont('Helvetica-Bold', 7)
        c.drawString(x + 11, top - 16, label)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 23)
        c.drawString(x + 11, top - 43, value)
        paragraph(detail, x + 11, top - 52, card_width - 20,
                  size=7.4, leading=10, color=MUTED, max_height=20)

    top -= 100
    top = section('What the results mean', top)
    top = paragraph(
        f'On <b>{main["n"]:,} unseen typical-market listings</b>, the model explains '
        f'<b>{main["r2"]*100:.1f}% of the variation</b> in advertised nightly prices. '
        'This measures explained variation, not a percentage of correct predictions. '
        f'Its predictions differ from the listed price by <b>${main["mae"]:.2f} per night '
        'on average</b>, counting errors in either direction. '
        f'The mean-price baseline misses by ${base["mae"]:.2f}; regression lowers '
        f'that average error by {data["mae_improvement_vs_mean_baseline_pct"]:.1f}%. '
        f'MSE is {main["mse"]:,.2f} USD squared; RMSE is ${main["rmse"]:.2f}.',
        margin, top, max_height=110)

    top -= 22
    top = section('Data and method', top)
    top = paragraph(
        f'The source contains <b>{data["dataset"]["rows"]:,} listings and '
        f'{data["dataset"]["columns"]} columns</b>. Fill '
        f'{data["missing_review_rates_filled_zero"]:,} missing review rates with zero '
        'because those listings have no reviews. Drop review dates, names, IDs and '
        'the 221-category neighbourhood field; remove 11 zero-price rows. '
        'Use borough, room type and seven numerical predictors. One-hot encoding '
        'drops the first category. A seeded 80/20 split (seed 42) precedes '
        'training-only 95th-percentile trimming and fitted preprocessing. '
        f'The retained market has <b>price &lt;= ${limits["price"]:.2f}</b> and '
        f'<b>minimum stay &lt;= {limits["minimum_nights"]:.0f} nights</b>: '
        f'{train["after_trimming"]:,} training and {test["after_trimming"]:,} test rows.',
        margin, top, max_height=130)

    top -= 22
    top = section('Cheap or expensive: where does it work better?', top)
    top = paragraph(
        f'<b>Inexpensive listings have lower dollar error:</b> MAE is '
        f'<b>${cheap["mae"]:.2f}</b> for prices up to '
        f'${bands["inexpensive_at_most"]:.0f}, versus <b>${expensive["mae"]:.2f}</b> '
        f'above ${bands["expensive_above"]:.0f}. These bands use training quartiles. '
        f'Expensive listings are underestimated by ${expensive["mean_residual"]:.2f} '
        'on average. Fitted values pull toward central prices; missing property '
        'size, amenities and nonlinear effects plausibly add to larger high-end misses. '
        f'Percentage errors differ: {cheap["mape_pct"]:.1f}% for inexpensive and '
        f'{expensive["mape_pct"]:.1f}% for expensive listings.',
        margin, top, max_height=115)

    top -= 22
    top = section('Business use and scope', top)
    top = paragraph(
        '<b>Use this as an initial pricing benchmark with human review.</b> '
        'The headline score applies to a test subset selected partly by observed '
        'price; an unknown future price cannot be screened the same way. '
        f'On all {full["n"]:,} positive-price test listings, including luxury and '
        f'long-stay cases, MAE is <b>${full["mae"]:.2f}</b> and R-squared is '
        f'<b>{full["r2"]:.4f}</b>. A random listing split can share hosts across sets '
        'and does not validate new hosts or future years. The 2019 snapshot omits '
        'key property details; coefficients are associations, not causal effects.',
        margin, top, max_height=115)

    footer_y = 54
    if top < footer_y + 18:
        raise ValueError(f'Report body overlaps footer: y={top:.1f}')
    c.setStrokeColor(colors.HexColor('#d6e3e9'))
    c.line(margin, footer_y + 7, page_w - margin, footer_y + 7)
    paragraph(
        'Data: Dgomonov, <link href="https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data" '
        'color="#176b87">New York City Airbnb Open Data (2019), Kaggle</link>; '
        '<link href="https://github.com/alexeygrigorev/datasets/blob/master/AB_NYC_2019.csv" '
        'color="#176b87">public CSV mirror</link>. '
        'Evidence: executed project notebook and output/metrics.json.',
        margin, footer_y, size=7, leading=9, color=MUTED, max_height=28)
    c.setFont('Helvetica', 7)
    c.setFillColor(MUTED)
    c.drawRightString(page_w - margin, 16, '1 / 1')
    c.showPage()
    c.save()
    return destination


if __name__ == '__main__':
    print(create_report())
