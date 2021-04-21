import os
import argparse
from pprint import pprint

import torch

from src.datasets import QuakeDataset
from src.models.custom_models import QuakeNet
from src.models.custom_configuration import QuakeNetConfig
from src.trainer import Trainer
from src.utils import get_num_params


def get_args():
    parser = argparse.ArgumentParser()

    # DATA PARAMS:
    parser.add_argument(
        "--train_filepath",
        type=str,
        required=True,
        help="Path to the train dataset.",
    )
    parser.add_argument(
        "--eval_filepath",
        type=str,
        required=True,
        help="Path to the eval dataset.",
    )
    parser.add_argument(
        "--data_columns",
        type=str,
        nargs="+",
        required=True,
        help="Columns to use in datasets.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        required=True,
        help="Grid size.",
    )
    parser.add_argument(
        "--num_points", type=int, required=True, help="Number of points per sample."
    )
    parser.add_argument(
        "--biggest_q_in",
        type=int,
        required=True,
        help="Biggest quake in `biggest_q_in` quakes.",
    )
    parser.add_argument(
        "--length_limit",
        type=int,
        default=None,
        required=False,
        help="Limit datasets sizes.",
    )

    # MODEL PARAMS:
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help=".",
    )

    # TRAINING PARAMS:
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help=".",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to evaluate during training.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=".",
    )

    # TODO input dim, seqlen, params from config

    return parser


def main():

    args = get_args().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print("ARGS:")
    pprint(vars(args))

    train_dataset = QuakeDataset(
        filepath=args.train_filepath,
        columns=args.data_columns,
        grid_x_size=args.grid_size,
        grid_y_size=args.grid_size,
        num_points=args.num_points,
        biggest_q_in=args.biggest_q_in,
        length_limit=args.length_limit,
    )
    print("Train dataset len: ", len(train_dataset))

    eval_dataset = QuakeDataset(
        filepath=args.eval_filepath,
        columns=args.data_columns,
        grid_x_size=args.grid_size,
        grid_y_size=args.grid_size,
        num_points=args.num_points,
        biggest_q_in=args.biggest_q_in,
        length_limit=args.length_limit,
    )

    print("Eval dataset len: ", len(eval_dataset))

    qn_config = QuakeNetConfig(
        num_points=args.num_points,
        x_size=args.grid_size,
        y_size=args.grid_size,
        encoder_points_reduction_dim=[1024, 256, 64],
    )
    model = QuakeNet.from_pretrained(args.model_name_or_path, config=qn_config)

    num_total, num_trainable = get_num_params(model)

    print("Number of total params in the model:", num_total)
    print("Number of trainable params in the model:", num_trainable)

    device = torch.device("cpu" if args.no_cuda else "cuda")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluate_during_training=args.evaluate_during_training,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_gradient_norm=args.max_norm,
        logging_steps=args.logging_steps,
        evaluate_steps=args.eval_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        fp16=False,
        device=device,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()
