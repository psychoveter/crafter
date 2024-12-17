import argparse
import os
import pandas as pd

if __name__ == '__main__':
    """
    Finds the best checkpoint in the folder
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=str, help='Path to run file')
    args = parser.parse_args()
    run = args.run

    assert os.path.exists(run)
    progress_file = os.path.join(run, 'progress.csv')
    assert os.path.exists(progress_file)

    progress = pd.read_csv(progress_file)
    sorted = progress.sort_values(by=['loss'], ascending=True)
    print(f"{run}/{sorted.values[0][2]}")