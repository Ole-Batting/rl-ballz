import wandb

from board import Board
from dql import DQL


def main(run, cfg=None):
    print("making env obj")
    env = Board(show_every=100)
    print("making trainer obj")
    if cfg is not None:
        trainer = DQL(**cfg)
    else:
        trainer = DQL()
    print("setup trainer with env")
    trainer.setup(env, logger=run)
    print("train")
    trainer.train(num_episodes=201)
    print("close trainer")
    trainer.close()
    run.finish()

if __name__ == "__main__":
    run = wandb.init(project="ballz-dql")
    main(run)