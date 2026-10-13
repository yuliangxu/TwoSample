#!/usr/bin/env python3
"""Validate a completed pooled-data analysis and build its ten-page final report.

Usage: python scripts/build_report.py RUN_DIRECTORY
Install requirements-report.txt first. The run directory must contain the staged
inputs, settings, source manifest, analysis outputs and ANALYSIS_COMPLETE marker.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import fitz
import reportlab

from report_layout import FIGURES, NAMES, Page, escape, figure_page, rows, sha
from prepare_run import verify_frozen_generation


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matrix(path, nrows, ncols, composition=False):
    with path.open(newline='') as stream:
        result = [[float(value) for value in row] for row in csv.reader(stream)]
    require(len(result) == nrows and all(len(row) == ncols for row in result),
            f'Unexpected dimensions: {path}')
    require(all(math.isfinite(value) for row in result for value in row),
            f'Nonfinite values: {path}')
    if composition:
        require(all(min(row) >= 0 and abs(math.fsum(row) - 1) < 5e-6 for row in result),
                f'Invalid compositional rows: {path}')
    return result


def validate(run):
    require((run / 'ANALYSIS_COMPLETE').is_file(), 'Analysis is not complete')
    manifest = json.loads((run / 'input_manifest.json').read_text())
    settings = json.loads((run / 'settings.json').read_text())
    require(settings['model_order'] == list(NAMES), 'Unexpected generator order')
    commit = (run / 'BATTS_commit.txt').read_text().strip()
    require(commit == manifest['BATTS_commit'] == settings['batts_commit'],
            'BATTS commit differs between settings and manifest')
    require(settings['boosting']['subsample_fraction'] == 1,
            'Final case study requires explicit full-sample boosting')
    expected_inputs = {'sample_train.csv': (1166, 123), 'sample_test.csv': (292, 123)}
    expected_inputs.update({f'sample_{key}.csv': (1000, 123) for key in NAMES})
    require(set(manifest['inputs']) == set(expected_inputs), 'Unexpected input set')
    for name, dimensions in expected_inputs.items():
        item = manifest['inputs'][name]
        require(sha(run / 'inputs' / name) == item['sha256'], f'Input hash changed: {name}')
        require(tuple(item['shape']) == dimensions, f'Input manifest shape mismatch: {name}')
        matrix(run / 'inputs' / name, *dimensions, composition=True)
    require(bool(manifest['source_scripts']), 'Source manifest is empty')
    for name, digest in manifest['source_scripts'].items():
        require(sha(run / name) == digest, f'Source hash changed: {name}')
    for name in ('build_report.py', 'report_layout.py', 'prepare_run.py'):
        snapshot = f'source/scripts/{name}'
        require(snapshot in manifest['source_scripts'], f'Report source not snapshotted: {name}')
        require(sha(Path(__file__).with_name(name)) == manifest['source_scripts'][snapshot],
                f'Report source differs from the staged analysis workflow: {name}')

    generation = verify_frozen_generation(run, manifest)

    stats = {}
    for row in rows(run / 'interval_summary.csv'):
        key = row['split'], row['generator'], row['source']
        require(key not in stats, f'Duplicate interval-summary key: {key}')
        n, cover = int(row['n']), int(row['contain_zero'])
        require(0 <= cover <= n, f'Invalid interval count: {key}')
        stats[key] = dict(n=n, cover=cover, exclude=n-cover,
                          mean=float(row['mean_abs_log_ratio']),
                          width=float(row['mean_interval_width']))
    expected_stats = {(split, key, source) for split in ('train', 'test')
                      for key in NAMES for source in ('Real', 'Generated')}
    require(set(stats) == expected_stats, 'Expected 16 interval-summary rows')
    table_hashes = {}
    for split, nreal in [('train', 1166), ('test', 292)]:
        for key in NAMES:
            name = f'revision/{split}/log_w_{key}.csv'
            table = matrix(run / name, nreal + 1000, 10)
            require(all(all(a <= b for a, b in zip(row[3:9], row[4:10])) for row in table),
                    f'Unordered posterior quantiles: {name}')
            table_hashes[name] = sha(run / name)
            for source, subset in [('Real', table[:nreal]), ('Generated', table[nreal:])]:
                item = stats[split, key, source]
                cover = sum(row[3] <= 0 <= row[9] for row in subset)
                mean = math.fsum(abs(row[2]) for row in subset) / len(subset)
                width = math.fsum(row[9] - row[3] for row in subset) / len(subset)
                require(item['n'] == len(subset) and item['cover'] == cover,
                        f'Summary counts disagree with estimator table: {name}, {source}')
                require(math.isclose(item['mean'], mean, rel_tol=1e-10, abs_tol=1e-12)
                        and math.isclose(item['width'], width, rel_tol=1e-10, abs_tol=1e-12),
                        f'Summary averages disagree with estimator table: {name}, {source}')

    null = rows(run / 'null/test_pointwise_credible_intervals.csv')
    require(len(null) == 292, 'Expected 292 held-out null intervals')
    null_cover = 0
    for index, row in enumerate(null, 1):
        values = [float(row[key]) for key in ('posterior_mean', 'ci_lower_025', 'ci_upper_975')]
        require(int(row['test_row']) == index and all(math.isfinite(value) for value in values)
                and values[1] <= values[2], f'Invalid null interval at row {index}')
        null_cover += values[1] <= 0 <= values[2]
    overall = [row for row in rows(run / 'null/coverage_summary.csv') if row['group'] == 'overall']
    require(len(overall) == 1 and int(overall[0]['n']) == 292
            and int(overall[0]['n_cover']) == null_cover, 'Null coverage summary disagrees')

    cv = rows(run / 'boosting_cv_summary.csv')
    expected_cv = {(split, key, estimator) for split in ('train', 'test')
                   for key in NAMES for estimator in ('gradient', 'hellinger')}
    require(len(cv) == 16 and {(row['split'], row['generator'], row['estimator'])
                             for row in cv} == expected_cv, 'Expected 16 unique boosting fits')
    max_trees = settings['boosting']['num_trees_max']
    for row in cv:
        selected = int(row['selected_trees'])
        require(1 <= selected <= max_trees and math.isfinite(float(row['minimum_cv_loss'])),
                'Invalid boosting selection summary')
        require(row['selected_at_search_limit'] == ('TRUE' if selected == max_trees else 'FALSE'),
                'Inconsistent boosting search-limit flag')
    analysis_validation = json.loads((run / 'analysis_validation.json').read_text())
    checks = dict(BATTS_commit=commit, BATTS_version=settings['batts_version'],
                  all_generators_trained_from_scratch=True,
                  independent_training_reproductions_verified=4,
                  generation_manifest_sha256=sha(run / 'generation/generation_manifest.json'),
                  model_order=list(NAMES), real_train=1166, real_test=292, taxa=123,
                  generated_per_model=1000, validated_comparisons=8,
                  validated_estimator_fits=24, validated_figures=7, null_intervals=292,
                  null_intervals_containing_zero=null_cover,
                  boosting_at_search_limit=sum(row['selected_at_search_limit'] == 'TRUE' for row in cv))
    for key, value in checks.items():
        require(analysis_validation[key] == value, f'Analysis validation disagrees: {key}')
    for number, _ in FIGURES:
        name = f'figure{number}_ggplot2.pdf'
        digest = hashlib.md5((run / 'figures' / name).read_bytes()).hexdigest()
        require(digest == analysis_validation['figure_md5'][name],
                f'Figure differs from the completed analysis: {name}')
    return dict(root=run, label='Original pooled data', train=1166, test=292, total=1458,
                taxa=123, commit=commit, version=settings['batts_version'], settings=settings,
                manifest=manifest, stats=stats, null_cover=null_cover,
                cap=sum(row['selected_at_search_limit'] == 'TRUE' for row in cv),
                expected={f'figures/figure{num}_ggplot2.pdf': sha(run / f'figures/figure{num}_ggplot2.pdf')
                          for num, _ in FIGURES}, table_hashes=table_hashes,
                analysis_validation=analysis_validation, generation=generation)


def overview(cfg):
    p = Page(cfg['label'], 1, 'Microbiome case study',
             'Full pooled data | Original preprocessing | ICFM, DT, D, MBGAN')
    p.para('<b>Two-sample Comparison through Additive Tree Models for Density Ratios</b>',
           12, 17, gap=13)
    p.metrics([('1,458', 'real observations'), ('123', 'retained taxa'),
               ('1,000', 'samples / generator')])
    p.heading('Final analysis')
    p.para('The original pooled real matrices contain <b>1,166 training and 292 test observations</b>. '
           'The original preprocessing, taxon order and observation split are preserved. '
           'All four generators are fitted from these training rows using released code and frozen '
           'recipes, in report order: <b>ICFM, DT, D, MBGAN</b>. No real test rows enter '
           'generator fitting or selection.')
    p.para(f"Eight real-versus-generated comparisons and the held-out null experiment use BATTS "
           f"<b>{escape(cfg['version'])}</b>, commit <b>{cfg['commit'][:8]}</b>. "
           'Gradient and Hellinger boosting explicitly use all fitting observations '
           'at each iteration (subsample_fraction = 1). Figures 6, 7 and S8-S12 summarize this run.')
    p.heading('Real-test pointwise posterior summaries')
    data = [['Generator', '95% intervals<br/>excluding zero',
             'Mean absolute<br/>posterior-mean log ratio']]
    for key, name in NAMES.items():
        item = cfg['stats']['test', key, 'Real']
        data.append([name, f"{item['exclude']}/{item['n']} ({100*item['exclude']/item['n']:.1f}%)",
                     f"{item['mean']:.3f}"])
    p.table(data, [143, 169, 204])
    p.para('These are descriptive pointwise posterior summaries, not global two-sample p-values. '
           'Intervals containing zero do not establish distributional equality. Each density-ratio '
           'comparison is fitted separately.', 9.5, 13)
    p.heading('Report contents')
    p.para('Page 2: methods and generator provenance.<br/>'
           'Pages 3-9: Figures 6, 7, S8, S9, S10, S11 and S12.<br/>'
           'Page 10: data sources, verification and reproducibility.', 9.5, 14)
    return p.finish()


def methods(cfg):
    p = Page(cfg['label'], 2, 'Methods and generator provenance',
             'Preserved real inputs; all generators fitted from the training rows')
    p.heading('Generator constructions')
    p.table([['Generator', 'Construction'],
             ['ICFM', 'CPU; random initialization; MLP with three hidden layers of width 64; 20,000 Adam updates, batch 256, rate 0.001, flow noise 0.0001; training seed 20261003; final update.'],
             ['DT', 'Positive-parameter Beta maximum-likelihood fits at 122 tree nodes, using the 1,166 training rows; seed 1.'],
             ['D', 'Dirichlet Newton maximum-likelihood fit on the 1,166 training rows; seed 1.'],
             ['MBGAN', 'GPU; random initialization; 500,000 generator iterations, five critic updates each, batch 32; RMSprop rate 0.00005; training seed 1; final iteration. Gradient penalty is differentiated through critic weights.']],
            [75, 441], 9)
    p.para('All four fits use only the preserved 1,166 training rows. Neural training starts from '
           'seeded random initialization; no historical weights are required. Training length and '
           'sampling settings are frozen before evaluation; the final update is used without '
           'test-based selection. Two full independent runs must agree within the recorded hardware '
           'and software environment; equality across CPU and GPU is not assumed.', 9.5, 13)
    bayes, boosting = cfg['settings']['bayes'], cfg['settings']['boosting']
    p.heading('BATTS settings')
    p.para(f"Seed {cfg['settings']['seed']} per estimator; {bayes['num_trees']} Bayesian trees; "
           f"{bayes['size_burnin']:,} burn-in and {bayes['size_backfitting']:,} retained iterations; "
           f"thinning {bayes['thin']}; fixed lambda {bayes['lambda_0']:g}; margin {bayes['margin_scale']:g}. "
           f"Boosting uses {boosting['K_CV']}-fold CV, up to {boosting['num_trees_max']:,} trees, "
           f"depth {boosting['max_resol']}, learning rate {boosting['learn_rate']:g}, "
           f"{boosting['n_bins']} bins, and subsample_fraction = 1. The figures show Bayesian results.",
           9.5, 13)
    p.heading('Training comparison: intervals excluding zero')
    data = [['Generator', 'Real training rows', 'Generated rows']]
    for key, name in NAMES.items():
        real, generated = [cfg['stats']['train', key, source] for source in ('Real', 'Generated')]
        data.append([name, f"{real['exclude']}/{real['n']}", f"{generated['exclude']}/{generated['n']}"])
    p.table(data, [172, 172, 172], 9)
    p.heading('Interpretation')
    p.para('The test comparisons fit density ratios using the real test rows themselves; they are '
           'not predictions from training-comparison estimators. Different reference sizes limit '
           'train/test comparisons. Filtering precedes the observation-level split; repeated '
           'observations may share participants across splits; participant-level separation is not established.', 9.5, 13)
    p.para(f"{cfg['cap']}/16 boosting fits selected the {boosting['num_trees_max']:,}-tree search limit. "
           'This follows the supplied single-chain workflow; no new multi-chain convergence '
           'assessment is claimed.', 9.5, 13)
    return p.finish()


def provenance(cfg):
    p = Page(cfg['label'], 10, 'Data sources and reproducibility',
             'Original preprocessing preserved through frozen real matrices')
    p.heading('Real-data composition')
    p.table([['Study', 'Training', 'Test', 'Total'], ['HMP_2019_ibdmdb', 751, 180, 931],
             ['HMP_2019_t2d', 210, 67, 277], ['HallAB_2017', 171, 38, 209],
             ['HanniganGD_2017', 34, 7, 41]], [258, 86, 86, 86], 8.5)
    p.para('The original curatedMetagenomicData selection is normalized over 123 retained taxa '
           'and stored at float32 precision. Counts refer to observations, not unique participants. '
           'The recovered sample-selection manifest, taxon order and preprocessing code accompany the release. '
           'Filtering is collection-wide, before the observation split. Real input hashes are verified.', 9.5, 13)
    p.heading('Execution and verification')
    p.para('All six input matrices are checked for dimensions, finite nonnegative values and '
           'row sums. All eight estimator tables are checked for dimensions, finite values, '
           'ordered posterior quantiles and agreement with numerical summaries. Seven source '
           'figures are embedded as PDF content. Every report page is rendered and checked for '
           'decoding warnings and text outside page bounds before COMPLETE is written. Generator '
           'training, sample hashes and reproduction evidence must also pass the provenance gate.', 9.5, 13)
    null = cfg['settings']['null']
    p.para(f"The separate null fit has <b>{cfg['null_cover']}/292</b> pointwise 95% intervals "
           f"containing zero. Balanced random real-training labels use seed {cfg['settings']['null_seed']}; "
           f"the fit uses {null['num_trees']} trees, {null['size_burnin']} burn-in and "
           f"{null['size_backfitting']} retained iterations, fixed lambda {null['lambda_0']:g} and "
           f"margin {null['margin_scale']:g}, evaluated on the real test matrix. One null experiment "
           'does not establish a repeated-sampling coverage guarantee.', 9.5, 13)
    p.heading('Portable run artifacts')
    p.para('Input and source hashes: input_manifest.json. Generator evidence: generation/. '
           'Fitting settings: settings.json and settings.R. '
           'Numerical results: interval_summary.csv and boosting_cv_summary.csv. Estimator tables: '
           'revision/. Figures: figures/. Verification: analysis_validation.json and report_manifest.json.',
           9, 13)
    p.para(f"<b>BATTS {escape(cfg['version'])}:</b> <font face=\"Courier\" size=\"8\">{cfg['commit']}</font><br/>"
           '<b>Workflow:</b> <link href="https://github.com/yuliangxu/TwoSample" color="#007C83">'
           'github.com/yuliangxu/TwoSample</link>. Workflow source snapshots and SHA-256 records are in the input manifest.',
           9, 13)
    p.heading('References for the real data')
    references = [
        ('Pasolli et al. (2017). Accessible, curated metagenomic data through ExperimentHub. Nature Methods 14:1023-1024.', '10.1038/nmeth.4468'),
        ('Lloyd-Price et al. (2019). Multi-omics of the gut microbial ecosystem in inflammatory bowel diseases. Nature 569:655-662.', '10.1038/s41586-019-1237-9'),
        ('Zhou et al. (2019). Longitudinal multi-omics of host-microbe dynamics in prediabetes. Nature 569:663-671.', '10.1038/s41586-019-1236-x'),
        ('Hall et al. (2017). A novel Ruminococcus gnavus clade enriched in inflammatory bowel disease patients. Genome Medicine 9:103.', '10.1186/s13073-017-0490-5'),
        ('Hannigan et al. (2018). Diagnostic Potential and Interactive Dynamics of the Colorectal Cancer Virome. mBio 9:e02248-18.', '10.1128/mBio.02248-18')]
    for citation, doi in references:
        p.para(escape(citation) + f' <link href="https://doi.org/{doi}" color="#007C83">doi:{doi}</link>',
               8, 11, gap=4)
    return p.finish()


def build(run):
    (run / 'COMPLETE').unlink(missing_ok=True)
    cfg = validate(run)
    fitz.TOOLS.mupdf_warnings()  # Clear warnings from earlier library use.
    document = fitz.open()
    document.insert_pdf(overview(cfg))
    document.insert_pdf(methods(cfg))
    contents = [[1, 'Overview and results', 1], [1, 'Methods and generator provenance', 2]]
    figures = []
    for page_no, (number, title) in enumerate(FIGURES, 3):
        page, evidence = figure_page(cfg, number, title, page_no)
        document.insert_pdf(page)
        page.close()
        figures.append(evidence)
        contents.append([1, f'Figure {number}: {title}', page_no])
    document.insert_pdf(provenance(cfg))
    contents.append([1, 'Data sources and reproducibility', 10])
    document.set_toc(contents)
    document.set_metadata(dict(title='Microbiome case study: full pooled data',
                              author='TwoSample case-study report',
                              subject='Original preprocessing; ICFM, DT, D, MBGAN; Figures 6, 7 and S8-S12',
                              creator='build_report.py; ReportLab and PyMuPDF'))
    output = run / 'case_study_report.pdf'
    partial = run / 'case_study_report.partial.pdf'
    document.save(partial, garbage=4, deflate=True)
    document.close()
    with fitz.open(partial) as check:
        require(len(check) == 10 and len(check.get_toc()) == 10, 'Expected ten report pages')
        outside = []
        for index, page in enumerate(check):
            page.get_pixmap(matrix=fitz.Matrix(.6, .6), alpha=False)
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        box = fitz.Rect(span['bbox'])
                        if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width+1 or box.y1 > page.rect.height+1:
                            outside.append([index + 1, span['text']])
        require(not outside, f'Text outside report pages: {outside}')
        for index, (number, _) in enumerate(FIGURES, 2):
            require(f'Figure {number}' in check[index].get_text()
                    and len(check[index].get_xobjects()) > 0, f'Missing embedded figure {number}')
    warnings = fitz.TOOLS.mupdf_warnings()
    require(not warnings, f'PDF decoding warnings: {warnings}')
    os.replace(partial, output)
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), pdf=output.name,
                  sha256=sha(output), pages=10, figures=figures,
                  BATTS_commit=cfg['commit'], BATTS_version=cfg['version'], model_order=list(NAMES),
                  input_manifest_sha256=sha(run / 'input_manifest.json'),
                  generation_manifest_sha256=sha(run / 'generation/generation_manifest.json'),
                  all_generators_trained_from_scratch=True,
                  independent_training_reproductions_verified=4,
                  settings_sha256=sha(run / 'settings.json'),
                  analysis_validation_sha256=sha(run / 'analysis_validation.json'),
                  analysis_complete_sha256=sha(run / 'ANALYSIS_COMPLETE'),
                  eight_estimator_tables_validated=True, estimator_tables=cfg['table_hashes'],
                  six_inputs_validated=True, interval_summaries_recomputed=True,
                  pdf_render_warnings=[], text_outside_pages=[],
                  null_intervals_containing_zero=cfg['null_cover'], boosting_at_search_limit=cfg['cap'],
                  reportlab_version=reportlab.Version, pymupdf_version=fitz.VersionBind,
                  builder_sha256=sha(__file__), layout_sha256=sha(Path(__file__).with_name('report_layout.py')))
    (run / 'report_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    summary = ['Final case study: full pooled data and original preprocessing', '',
               'Real inputs: 1166 training x 123 taxa; 292 test x 123 taxa. Generators: 1000 x 123 each.',
               'Model order: ICFM, DT, D, MBGAN.',
               'All four generators are fitted from the 1166 training rows; neural models start from random initialization.',
               'Training recipes are frozen; no real test rows enter generator training or selection.',
               f"BATTS {cfg['version']} ({cfg['commit']}); boosting subsample_fraction=1.", '',
               'Real-test intervals excluding zero and mean absolute posterior-mean log ratio:']
    for key, name in NAMES.items():
        item = cfg['stats']['test', key, 'Real']
        summary.append(f"{name}: {item['exclude']}/{item['n']}; {item['mean']:.6f}")
    summary += ['', f"Null intervals containing zero: {cfg['null_cover']}/292.",
                f"Boosting fits at the search limit: {cfg['cap']}/16.",
                'Test comparisons fit density ratios using the real test rows themselves.',
                'Pointwise intervals do not establish global equality or general coverage guarantees.',
                '', 'PDF: case_study_report.pdf (10 pages; seven case-study figures).']
    (run / 'RESULTS.txt').write_text('\n'.join(summary) + '\n')
    # A completed report attests to this analysis and this exact output PDF.
    (run / 'COMPLETE').write_text(json.dumps(dict(report_sha256=sha(output),
                                               report_manifest_sha256=sha(run / 'report_manifest.json'),
                                               completed_utc=record['created_utc']), indent=2) + '\n')
    print('\n'.join(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Completed analysis directory')
    build(parser.parse_args().run.resolve())
