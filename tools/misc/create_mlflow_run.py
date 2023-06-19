import argparse
import sys

import mlflow


def parse_args():
    parser = argparse.ArgumentParser('Create run of mlflow')
    parser.add_argument(
        '--exp-name',
        type=str,
        default='',
        required=True,
        help='mlflow experiment name')
    parser.add_argument(
        '--run-name', type=str, default=None, help='mlflow run name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mlflow.set_experiment(args.exp_name)
    run_id = mlflow.start_run(run_name=args.run_name).info.run_id
    sys.stdout.write(run_id)
    sys.exit(0)


if __name__ == '__main__':
    main()
