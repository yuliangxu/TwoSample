"""Reusable page and vector-figure layout for the final pooled case study."""
import csv
import hashlib
import io
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle

NAVY = '#17324D'
TEAL = '#007C83'
INK = '#253544'
GRAY = '#5D6D79'
PALE = '#EFF5F7'
LINE = '#D5E0E5'
FIGURES = [('6', 'Test-data ordination'), ('7', 'Test-data density ratios'),
           ('S8', 'Training-data ordination'), ('S9', 'Training-data density ratios'),
           ('S10', 'Training posterior mean and quantiles'),
           ('S11', 'Test posterior mean and quantiles'), ('S12', 'Held-out null diagnostic')]
NAMES = {'icfm': 'ICFM', 'dt': 'DT', 'd': 'D', 'mbgan': 'MBGAN'}


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


class Page:
    def __init__(self, label, number, title, subtitle='', size=(612, 792)):
        self.w, self.h = size
        self.buf = io.BytesIO()
        self.c = canvas.Canvas(self.buf, pagesize=size, invariant=1)
        self.y = 46
        self.x = 48
        self.width = self.w - 96
        self.c.setFillColor(colors.HexColor(TEAL))
        self.c.rect(0, self.h - 7, self.w, 7, fill=1, stroke=0)
        self.para(title, 22 if number in (1, 2, 10) else 18, 27, bold=True, gap=6)
        if subtitle:
            self.para(subtitle, 9, 13, color=GRAY, gap=12)
        self.c.setStrokeColor(colors.HexColor(LINE))
        self.c.line(48, 38, self.w - 48, 38)
        self.c.setFont('Helvetica', 8)
        self.c.setFillColor(colors.HexColor(GRAY))
        self.c.drawString(48, 23, label + '  |  Final case study')
        self.c.drawRightString(self.w - 48, 23, str(number))

    def para(self, text, size=10, leading=14, bold=False, color=INK, gap=8,
             x=None, width=None):
        x = self.x if x is None else x
        width = self.width if width is None else width
        p = Paragraph(text, ParagraphStyle('body', fontName='Helvetica-Bold' if bold else 'Helvetica',
                      fontSize=size, leading=leading, textColor=colors.HexColor(color)))
        _, height = p.wrap(width, 2000)
        if self.y + height > self.h - 52:
            raise ValueError(f'Page overflow: {text[:70]} at {self.y} + {height} / {self.h}')
        p.drawOn(self.c, x, self.h - self.y - height)
        self.y += height + gap

    def heading(self, text):
        self.y += 3
        self.para(text, 12, 16, bold=True, color=NAVY, gap=6)

    def table(self, data, widths, fontsize=9, gap=11):
        style = ParagraphStyle('cell', fontName='Helvetica', fontSize=fontsize,
                               leading=fontsize + 3, textColor=colors.HexColor(INK))
        head = ParagraphStyle('head', parent=style, fontName='Helvetica-Bold',
                              textColor=colors.white)
        formatted = [[Paragraph(str(cell), head if i == 0 else style) for cell in row]
                     for i, row in enumerate(data)]
        t = Table(formatted, colWidths=widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor(PALE), colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor(LINE))]))
        _, height = t.wrap(self.width, 2000)
        if self.y + height > self.h - 52:
            raise ValueError(f'Table overflow: {self.y} + {height}')
        t.drawOn(self.c, self.x, self.h - self.y - height)
        self.y += height + gap

    def metrics(self, values):
        w = self.width / len(values)
        for i, (value, label) in enumerate(values):
            x = 48 + i * w
            self.c.setFillColor(colors.HexColor(PALE))
            self.c.rect(x, self.h - self.y - 56, w - 5, 56, stroke=0, fill=1)
            self.c.setFillColor(colors.HexColor(TEAL))
            self.c.setFont('Helvetica-Bold', 21)
            self.c.drawString(x + 10, self.h - self.y - 24, value)
            self.c.setFont('Helvetica', 9)
            self.c.setFillColor(colors.HexColor(INK))
            self.c.drawString(x + 10, self.h - self.y - 43, label)
        self.y += 71

    def finish(self):
        self.c.showPage()
        self.c.save()
        return fitz.open(stream=self.buf.getvalue(), filetype='pdf')

def figure_page(cfg, num, title, page_number):
    path = cfg['root'] / f'figures/figure{num}_ggplot2.pdf'
    digest = sha(path)
    if digest != cfg['expected'][f'figures/{path.name}']:
        raise ValueError(f'Checksum mismatch: {path}')
    src = fitz.open(path)
    if len(src) != 1:
        raise ValueError(f'Expected a single-page figure: {path}')
    sw, sh = src[0].rect.width, src[0].rect.height
    size = (612, 792) if num in ('7', 'S9') else (sw + 96, sh + 226)
    p = Page(cfg['label'], page_number, f'Figure {num}  |  {title}',
             cfg['label'] + f"  |  BATTS {cfg['commit'][:8]}", size)
    top = 105
    rect = fitz.Rect((p.w-sw)/2, top, (p.w+sw)/2, top+sh)
    p.y = top + sh + 15
    train = num in ('S8', 'S9', 'S10')
    n = cfg['train'] if train else cfg['test']
    split = 'training' if train else 'test'
    if num in ('6', 'S8'):
        caption = (f'<b>Real {split} compositions versus four generators.</b> Each panel contains '
                   f'{n:,} real observations (blue dots) and 1,000 generated observations (orange crosses). '
                   'The first two principal coordinates use Bray-Curtis dissimilarities with an additive '
                   'correction. Each panel is a separate joint ordination of real and generated data; '
                   'axes are not shared across generator panels. Visual overlap in two dimensions '
                   'does not establish equality of the full compositional distributions.')
    elif num in ('7', 'S9'):
        caption = (f'<b>Bayesian density-ratio summaries for the {split} comparisons.</b> '
                   '(a) Posterior mean log density ratio on the corrected Bray-Curtis ordination. '
                   '(b) Classification of central 95% intervals as above zero, covering zero, or below zero. '
                   '(c) Posterior means and intervals at 20 generated-row indices selected by the '
                   'average absolute posterior mean across the four generator comparisons. These are '
                   'selected extremes. Row labels refer to the stacked real/generated matrices. '
                   'Equal display indices across generators represent independent generated samples, '
                   'not matched biological observations.')
    elif num in ('S10', 'S11'):
        caption = (f'<b>Posterior summaries across the {split} compositions.</b> Columns correspond to '
                   'ICFM, DT, D and MBGAN, in that order. Rows show the posterior mean, lower '
                   '2.5% quantile and upper 97.5% quantile of the log density ratio. Real samples are dots; '
                   'generated samples are crosses. These grids use classical Bray-Curtis PCoA without '
                   'an additive correction, following the legacy workflow, and a symmetric-log color scale. '
                   'Their coordinates differ from those in Figures 6, 7, S8 and S9. '
                   f"The plotted color limits are {'-30 to 30' if train else '-12 to 12'}.")
    else:
        null = cfg['settings']['null']
        caption = (f'<b>Real-versus-real null experiment.</b> Balanced random labels are assigned to '
                   f'the {cfg["train"]:,} real training observations (seed {cfg["settings"]["null_seed"]}). '
                   f'The Bayesian fit uses {null["num_trees"]} trees, {null["size_burnin"]} burn-in and '
                   f'{null["size_backfitting"]} retained iterations, fixed lambda {null["lambda_0"]:g} '
                   f'and margin {null["margin_scale"]:g}. '
                   f'The {cfg["test"]} held-out real test observations are ordered by posterior mean. '
                   f'{cfg["null_cover"]}/{cfg["test"]} pointwise 95% credible intervals contain zero. '
                   'This is the observed result of one null experiment, not a guarantee of nominal coverage.')
    p.para(caption, 9.5 if num in ('7', 'S9') else 10, 13.5, gap=6)
    doc = p.finish()
    doc[0].show_pdf_page(rect, src, 0)
    src.close()
    return doc, dict(figure=num, report_page=page_number, source=f'figures/{path.name}', sha256=digest,
                     source_points=[sw, sh], embedded_points=list(rect))
