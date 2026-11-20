"""Summarize route and required-element failures from a completed regression."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path

from .run import checksum


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_index(data):
    rows={}
    for split in ('development','internal_test'):
        for row in read_jsonl(data/f'{split}.jsonl'):
            rows[row['id']]=row
    return rows


def analyze(regression,data):
    sources=source_index(data)
    details=[]
    summaries={}
    missing_by_capability=defaultdict(Counter)
    missing_by_family=defaultdict(Counter)
    for path in sorted(regression.glob('*.jsonl')):
        mode=path.stem
        rows=read_jsonl(path)
        route_failures=[]
        element_failures=[]
        for row in rows:
            source=sources[row['id']]
            missing=[value for value in row['required_checks']
                     if value not in row['matched_checks']]
            if not row['route_correct']:
                route_failures.append(row['id'])
            if missing:
                element_failures.append(row['id'])
                missing_by_capability[source['capability']].update(missing)
                missing_by_family[source['behavior_family']].update(missing)
            if not row['route_correct'] or missing:
                details.append({
                    'split_mode':mode,'id':row['id'],
                    'capability':source['capability'],
                    'behavior_family':source['behavior_family'],
                    'expected_capability':row['expected_capability'],
                    'routed_capability':row['routed_capability'],
                    'route_correct':row['route_correct'],
                    'response_source':row['response_source'],
                    'missing_checks':missing,
                    'matched_checks':row['matched_checks'],
                    'answer':row['answer'],
                })
        summaries[mode]={
            'count':len(rows),'route_failures':len(route_failures),
            'route_failure_ids':route_failures,
            'minimum_element_failures':len(element_failures),
            'minimum_element_failure_ids':element_failures,
        }
    return {
        'kind':'petlm_routing_failure_analysis_v1',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'regression':str(regression),'data':str(data),
        'regression_report_sha256':checksum(regression/'report.json'),
        'data_manifest_sha256':checksum(data/'manifest.json'),
        'summaries':summaries,
        'missing_checks_by_capability':{
            key:dict(value.most_common()) for key,value in sorted(missing_by_capability.items())},
        'missing_checks_by_behavior_family':{
            key:dict(value.most_common()) for key,value in sorted(missing_by_family.items())},
        'failure_details':details,
        'limitations':('Required-element matching is literal substring analysis. '
                       'It identifies regression targets but is not semantic human scoring.'),
    }


def markdown(report):
    lines=['# 开发集与内部测试集失败分析','',
           f"生成时间：`{report['created_at']}`",'',
           '## 汇总','',
           '| 数据 | 条数 | 路由错误 | 最低要素缺失 |','|---|---:|---:|---:|']
    for name,value in report['summaries'].items():
        lines.append(f"| {name} | {value['count']} | {value['route_failures']} | {value['minimum_element_failures']} |")
    lines.extend(['','## 路由错误',''])
    route=[row for row in report['failure_details'] if not row['route_correct']]
    if not route:
        lines.append('当前回归未发现路由错误。')
    else:
        for row in route:
            lines.append(f"- `{row['id']}`：预期 `{row['expected_capability']}`，实际 `{row['routed_capability']}`。")
    lines.extend(['','## 缺失要素热点',''])
    for capability,counts in report['missing_checks_by_capability'].items():
        common='、'.join(f'`{key}` {value} 次' for key,value in list(counts.items())[:10])
        lines.append(f'- `{capability}`：{common or "无"}')
    lines.extend(['','## 使用边界','',report['limitations'],
                  '本报告只使用开发集和内部测试集；不得据此读取或调整 v3 冻结盲测答案。',''])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--regression',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    report=analyze(args.regression,args.data)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'REPORT.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='failure_details'},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
