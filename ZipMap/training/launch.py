# Code for ZipMap (CVPR 2026); created by Haian Jin

import argparse
from hydra import initialize, compose
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    parser.add_argument("overrides", nargs="*", help="Overrides to the config")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)
    
    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


