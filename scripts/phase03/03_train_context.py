"""Run one bounded, predeclared CPU context condition."""
import argparse
from storyseed.context_training import train_condition

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, choices=[1, 8], required=True)
    args = parser.parse_args()
    train_condition(args.context)
