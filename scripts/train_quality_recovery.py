import argparse
from storyseed.quality_training import pilot, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pilot or run the declared quality-recovery scratch training.')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    pilot() if args.pilot else train()
