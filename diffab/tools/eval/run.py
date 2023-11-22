import os
import argparse
import ray
import shelve
import time
import pandas as pd
from typing import Mapping

from diffab.tools.eval.base import EvalTask, TaskScanner
from diffab.tools.eval.energy import eval_interface_energy, eval_pred_ddg
from diffab.tools.eval.hydropathy import eval_hydropathy
from diffab.tools.eval.similarity import eval_similarity


@ray.remote(num_cpus=1)
def evaluate(task, args):
    funcs = []
    if not args.no_similarity:
        funcs.append(eval_similarity)
    if not args.no_energy:
        funcs.append(eval_interface_energy)
    if not args.no_hydro:
        funcs.append(eval_hydropathy)
    if not args.no_pred_ddg:
        funcs.append(eval_pred_ddg)
    for f in funcs:
        task = f(task)
    return task


def dump_db(db: Mapping[str, EvalTask], path):
    table = []
    for task in db.values():
        if 'abopt' in path and task.scores['seqid'] >= 100.0:
            # In abopt (Antibody Optimization) mode, ignore sequences identical to the wild-type
            continue
        table.append(task.to_report_dict())
    table = pd.DataFrame(table)
    table.to_csv(path, index=False, float_format='%.6f')
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='summary.csv')
    parser.add_argument('--pfx', type=str, default=None)
    parser.add_argument('--no_similarity', action='store_true', default=False)
    parser.add_argument('--no_energy', action='store_true', default=False)
    parser.add_argument('--no_hydro', action='store_true', default=False)
    parser.add_argument('--no_pred_ddg', action='store_true', default=False)
    args = parser.parse_args()
    ray.init(num_cpus=15)
    # ray.init(num_cpus=1)

    db_path = os.path.join(args.root, 'evaluation_db')
    with shelve.open(db_path) as db:
        scanner = TaskScanner(root=args.root, postfix=args.pfx, db=db)

        # while True:
        tasks = scanner.scan()
        futures = [evaluate.remote(t, args) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            for done_id in done_ids:
                done_task = ray.get(done_id)
                done_task.save_to_db(db)
                print(f'Remaining {len(futures)}. Finished {done_task.in_path}')
            db.sync()

        dump_db(db, os.path.join(args.root, args.output_file))
        time.sleep(1.0)

if __name__ == '__main__':
    main()
