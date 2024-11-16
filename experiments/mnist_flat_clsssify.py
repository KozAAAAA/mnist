from argparse import ArgumentParser

import ai


def main(args):
    # todo: load mnist dataset
    train_ds, val_ds = ...

    # todo: create and optimize model (add regularization like dropout and batch normalization)
    model = ...

    # todo: create optimizer (optional: try with learning rate decay)
    optimizer = ...

    # todo: define query function
    def query():
        pass

    # todo: define train function
    def train():
        pass

    # todo: run training and evaluation for number or epochs (from argument parser)
    #  and print results (accumulated) from each epoch (train and val separately)
    ...


if __name__ == '__main__':
    parser = ArgumentParser()
    # todo: pass arguments
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        ai.utils.allow_memory_growth()

    main(args)
