"""Train one predeclared condition in an isolated foreground process."""
import argparse
from storyseed.experiment_data import protocol
from storyseed.experiment_training import train_condition


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=['main128','context64'],required=True)
    args=parser.parse_args()
    condition=next(value for value in protocol()['conditions'] if value['id']==args.condition)
    train_condition(condition)


if __name__=='__main__':main()
