"""Create comparison tables from an immutable, previously audited snapshot.

This standard-library report does no sampling and does not re-audit predictions.
Incomplete PDEs never receive a macro score. The all66 layout is available for
the Burgers phase, but must be exercised on its real results before closeout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from cocogen_eval.collect_metrics import PDES, require, sha_file
from cocogen_eval.common import write_json


NAMES = dict(darcy='Darcy', poisson='Poisson', helmholtz='Helmholtz',
             nsnonbounded='NS', burger='Burgers')
SETTINGS = dict(full_forward='完整观测正问题', full_inverse='完整观测反问题',
                sparse_forward='稀疏正问题', sparse_inverse='稀疏反问题',
                sparse_joint='稀疏联合重建', random='随机 500 点',
                sensor_column='五个固定空间位置的时间轨迹')
DISTS = ('id', 'smooth', 'rough')
MODELS = ('cocogen', 'fm4pde')


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), 'Inconsistent aggregate')


def aggregate(rows):
    require(bool(rows), 'Cannot aggregate an empty group')
    means = {model: statistics.fmean(row[model] for row in rows) for model in MODELS}
    return dict(cells=len(rows), **means,
                ratio=means['cocogen']/means['fm4pde'] if means['fm4pde'] else None,
                smaller_mean_cells=sum(row['cocogen'] < row['fm4pde'] for row in rows),
                sampler_gpu_hours=sum(row['sampler_seconds'] for row in rows)/3600)


def summarize(metrics_path, coverage_path, allow_partial=False):
    metrics = json.loads(metrics_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    require(metrics['phase'] in ('first60', 'all66'), 'Unknown evaluation phase')
    pdes = PDES + (('burger',) if metrics['phase'] == 'all66' else ())
    expected_count = 66 if 'burger' in pdes else 60
    require(metrics['expected_cells'] == expected_count, 'Unexpected evaluation scope')
    require(metrics['status'] == 'complete' or allow_partial, 'A complete snapshot is required')
    cells = {row['cell']: row for row in metrics['cells']}
    require(len(cells) == len(metrics['cells']) == metrics['completed_cells'], 'Duplicate cells')
    require(coverage['status'] == 'passed' and coverage['metrics_sha256'] == sha_file(metrics_path),
            'Audit coverage does not bind this snapshot')
    require(set(coverage['cells']) == set(cells) and coverage['verified_cells'] == len(cells)
            and coverage['verified_samples'] == len(cells)*1000, 'Incomplete audit coverage')
    fields = defaultdict(list)
    for row in metrics['fields']:
        require(row['cell'] in cells and row['n'] == 1000, 'Invalid field membership')
        require(all(math.isfinite(row[model]) and row[model] >= 0 for model in MODELS), 'Invalid error')
        fields[row['cell']].append(row)
    for name, row in cells.items():
        require(row['n'] == 1000 and row['pde'] in pdes, 'Invalid cell scope')
        require(sorted(field['field'] for field in fields[name]) == sorted(row['target_fields']),
                'Wrong or duplicate target fields')
        for model in MODELS:
            close(row[model], statistics.fmean(field[model] for field in fields[name]))
    reference_pdes = {row['pde']: row for row in metrics['pdes']}
    require(set(reference_pdes) == set(pdes), 'Wrong PDE coverage')
    summaries = []
    for pde in pdes:
        subset = [row for row in cells.values() if row['pde'] == pde]
        settings = ('random', 'sensor_column') if pde == 'burger' else tuple(list(SETTINGS)[:5])
        expected = {f'{pde}/{dist}/{setting}' for dist in DISTS for setting in settings}
        require({row['cell'] for row in subset} <= expected, 'Unexpected cell identity')
        complete = len(subset) == len(expected)
        reference = reference_pdes[pde]
        require(reference['completed_cells'] == len(subset) and reference['expected_cells'] == len(expected),
                'Incorrect PDE counts')
        require((reference['status'] == 'complete') == complete, 'Incorrect PDE completion flag')
        totals = aggregate(subset) if complete else None
        for model in MODELS:
            if complete:
                close(totals[model], reference[f'{model}_cell_macro'])
            else:
                require(reference[f'{model}_cell_macro'] is None, 'Partial PDE must have no macro')
        summaries.append(dict(pde=pde, completed_cells=len(subset), expected_cells=len(expected),
            macro=totals,
            by_distribution={dist: aggregate([row for row in subset if row['dist'] == dist])
                             for dist in DISTS} if complete else None,
            by_setting={setting: aggregate([row for row in subset if row['setting'] == setting])
                        for setting in settings} if complete else None))
    complete = all(row['macro'] is not None for row in summaries)
    require((metrics['status'] == 'complete') == complete, 'Incorrect phase completion flag')
    overall = {model: statistics.fmean(row['macro'][model] for row in summaries)
               for model in MODELS} if complete else None
    if complete:
        require(not metrics['pending_cells'] and len(cells) == expected_count, 'Incomplete final scope')
        for model in MODELS:
            close(overall[model], metrics['overall_pde_macro'][model])
    else:
        require(metrics['overall_pde_macro'] is None, 'Partial phase must have no macro')
    return dict(status=metrics['status'], phase=metrics['phase'], as_of=metrics['generated_at'],
        final_study_complete=False, completed_cells=len(cells), expected_cells=expected_count,
        overall_pde_macro=overall, pdes=summaries, cells=metrics['cells'], fields=metrics['fields'],
        observed_sampler_gpu_hours=sum(row['sampler_seconds'] for row in cells.values())/3600,
        source=dict(metrics_file=metrics_path.name, metrics_sha256=sha_file(metrics_path),
                    coverage_file=coverage_path.name, coverage_sha256=sha_file(coverage_path),
                    reporter_sha256=sha_file(__file__), raw_predictions_reloaded=False),
        weighting=metrics['weighting'], metric=metrics['metric'], comparison=metrics['comparison'])


def render(report):
    names = '四模型' if report['phase'] == 'first60' else '五模型'
    lines = [f'# CoCoGen 与 FM4PDE：{names}主实验比较', '',
        f"数据截至 `{report['as_of']}`。已完成并有归档复核覆盖 **{report['completed_cells']}/{report['expected_cells']} 个单元**，每单元 1,000 例。", '']
    if report['overall_pde_macro']:
        values = report['overall_pde_macro']
        lines += [f"全部 PDE 等权宏平均相对 L2：**CoCoGen {values['cocogen']:.2%}，FM4PDE {values['fm4pde']:.2%}**。", '']
    else:
        lines += ['尚未产生本阶段总体宏平均；未完成 PDE 的汇总留空。', '']
    lines += ['## 已完成 PDE 的比较', '', '相对 L2 越低越好。“胜出单元”表示该单元的目标场平均误差更低，不是统计显著性或逐样本胜率。', '',
        '| PDE | 完成单元 | CoCoGen | FM4PDE | 误差比 | 胜出单元 | 采样 GPU 小时 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in report['pdes']:
        prefix = f"| {NAMES[row['pde']]} | {row['completed_cells']}/{row['expected_cells']} |"
        value = row['macro']
        if value:
            ratio = f"{value['ratio']:.3f}" if value['ratio'] is not None else '—'
            lines.append(f"{prefix} {value['cocogen']:.2%} | {value['fm4pde']:.2%} | {ratio} | {value['smaller_mean_cells']}/{value['cells']} | {value['sampler_gpu_hours']:.3f} |")
        else:
            lines.append(f'{prefix} — | — | — | — | — |')
    lines += ['', '## 各任务平均', '', '每行对同一 PDE 的三个分布等权；联合任务先对 a/u 等权，只占一个任务权重。未完成的 PDE 不进入本表。', '',
        '| PDE | 任务 | CoCoGen | FM4PDE | 胜出单元 |', '|---|---|---:|---:|---:|']
    for row in report['pdes']:
        for setting, value in (row['by_setting'] or {}).items():
            lines.append(f"| {NAMES[row['pde']]} | {SETTINGS[setting]} | {value['cocogen']:.2%} | {value['fm4pde']:.2%} | {value['smaller_mean_cells']}/{value['cells']} |")
    lines += ['', '## 各分布平均', '', '每行对同一 PDE、同一分布的全部任务等权。', '',
        '| PDE | 分布 | CoCoGen | FM4PDE | 胜出单元 |', '|---|---|---:|---:|---:|']
    for row in report['pdes']:
        for dist, value in (row['by_distribution'] or {}).items():
            lines.append(f"| {NAMES[row['pde']]} | {dist} | {value['cocogen']:.2%} | {value['fm4pde']:.2%} | {value['smaller_mean_cells']}/{value['cells']} |")
    lines += ['', '## 全部已完成单元的目标场', '',
        '| PDE | 分布 | 任务 | 目标场 | CoCoGen | FM4PDE | CoCoGen 逐例误差更低比例 |',
        '|---|---|---|---|---:|---:|---:|']
    for row in report['fields']:
        lines.append(f"| {NAMES[row['pde']]} | {row['dist']} | {SETTINGS[row['setting']]} | {row['field']} | {row['cocogen']:.2%} | {row['fm4pde']:.2%} | {row['smaller_error_fraction']:.1%} |")
    lines += ['', '## 口径与适用范围', '',
        '- 在物理空间逐例计算完整目标场的相对 L2，再对 1,000 例取均值。双方使用相同归档样本编号及有效观测，各自使用冻结采样配置。',
        '- 正问题评价 u，反问题评价 a，联合重建对 a/u 等权。Burgers 仅评价 u；若包含 Burgers，先算各 PDE 的宏平均，再对五个 PDE 等权，不能直接平均 66 个单元。',
        '- a 在 Darcy 中为系数，在 Poisson/Helmholtz 中为源项，在 NS 中为初始涡量。NS 的物理修正使用两端点近似残差。',
        '- 同一分布的任务复用样本，一个固定采样种子；单元数与逐样本比较不能当作独立重复实验或多种子置信度。',
        '- 本表描述既有权重和已冻结适配的表现，不能单独识别训练、采样预算或条件方式的因果贡献。',
        f"- 已完成单元累计纯采样 {report['observed_sampler_gpu_hours']:.3f} GPU 小时，是各卡时间之和；不代表墙钟时间，不含训练、校准和归档。", '',
        '## 可追溯来源', '',
        f"- 指标快照：`{report['source']['metrics_file']}`，SHA256 `{report['source']['metrics_sha256']}`。",
        f"- 归档复核覆盖：`{report['source']['coverage_file']}`，SHA256 `{report['source']['coverage_sha256']}`。",
        '- 汇总脚本核对快照哈希、覆盖范围和各级权重；本次制表没有重新加载预测。原始数值复核见覆盖文件引用的审计收据。',
        '- 本表完成不等于全研究归档完成；训练质量、环境恢复和清理需另行核验。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = summarize(args.metrics, args.coverage, args.allow_partial)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'comparison.json', result)
    (args.output/'comparison.md').write_text(render(result))
    print(json.dumps(dict(status=result['status'], completed_cells=result['completed_cells'],
                          output=str(args.output))))
