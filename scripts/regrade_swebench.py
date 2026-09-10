"""Regrade saved predictions for unscored tasks without running a model."""
import argparse
import asyncio
import json
from pathlib import Path

from datasets import load_dataset
from uni_agent.tasks import TaskConfigResolver
from coding_opd.swe_bench_task import SWEBenchTask, SWEBenchTaskConfig


async def main(args):
    result = json.loads(args.result.read_text())
    resolver = TaskConfigResolver.from_file(result['task_config'])
    samples = {s['extra_info']['tools_kwargs']['task']['metadata']['instance_id']: s
               for s in load_dataset('parquet', data_files=result['data_path'], split='train')}
    predictions = {}
    for path in args.predictions.glob('*/verification/prediction.json'):
        predictions[json.loads(path.read_text())['instance_id']] = path
    for record in result['tasks']:
        tid = record['task_id']
        if record['scored_sessions'] or tid not in predictions:
            continue
        sample = samples[tid]
        config = resolver.resolve(sample['extra_info']['tools_kwargs']['task'])
        config.update(prediction_path=str(predictions[tid]),
                      verification_dir=str(args.output.parent / 'regrade' / tid),
                      eval_timeout=args.eval_timeout)
        config['agent']['model']['model_name'] = result['served_model_name']
        try:
            outcome = await SWEBenchTask(SWEBenchTaskConfig(**config)).run()
            record.update(score=outcome.reward, session_scores=[outcome.reward],
                          scored_sessions=1, status='finished')
            print(tid, 'score', outcome.reward, flush=True)
        except Exception as exc:
            print(tid, type(exc).__name__, str(exc), flush=True)
        result['scored_sessions'] = sum(t['scored_sessions'] for t in result['tasks'])
        result['completed_tasks'] = sum(t['scored_sessions'] >= result['n'] for t in result['tasks'])
        result['mean_score'] = sum(t['score'] for t in result['tasks']) / result['num_tasks']
        result['regrade'] = {'source': str(args.result), 'eval_timeout': args.eval_timeout,
                             'predictions': str(args.predictions)}
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval-timeout', type=float, default=3600)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('output already exists; use a new recovery directory')
    asyncio.run(main(args))
