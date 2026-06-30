import argparse
import sys

from src.dataset import build_data_pipelines
from src.infer import execute_inference
from src.optimize import run_optuna_study
from src.tester import perform_sanity_check
from src.trainer import execute_training_loop


def run_main_dispatcher():
    argument_parser = argparse.ArgumentParser(description="Flora ML Pipeline")
    argument_parser.add_argument("--mode", type=str, required=True, choices=["test", "train", "optimize", "infer"])
    argument_parser.add_argument("--train_directory", type=str, default="dataset/plantnet_300K/images/train")
    argument_parser.add_argument("--validation_directory", type=str, default="dataset/plantnet_300K/images/val")
    argument_parser.add_argument("--batch_size", type=int, default=32)
    argument_parser.add_argument("--learning_rate", type=float, default=1e-3)
    argument_parser.add_argument("--weight_decay", type=float, default=1e-1)
    argument_parser.add_argument("--num_epochs", type=int, default=40)
    argument_parser.add_argument("--loss_type", type=str, default="cross_entropy", choices=["cross_entropy", "focal"])
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument("--resume_directory", type=str, default=None)
    argument_parser.add_argument("--run_directory", type=str, default=None)
    argument_parser.add_argument("--image_path", type=str, default=None)

    parsed_arguments, unknown_arguments = argument_parser.parse_known_args()

    dynamic_hyperparameters_dictionary = {}
    for argument_string in unknown_arguments:
        if argument_string.startswith("--"):
            splitted_argument = argument_string.split("=")
            key_name = splitted_argument[0].lstrip("-")
            if len(splitted_argument) > 1:
                value_data = splitted_argument[1]
            else:
                value_data = True
            dynamic_hyperparameters_dictionary[key_name] = value_data

    if parsed_arguments.mode in ["test", "train", "optimize"]:
        train_dataloader, validation_dataloader, train_dataset, validation_dataset = build_data_pipelines(
            train_directory_path=parsed_arguments.train_directory,
            validation_directory_path=parsed_arguments.validation_directory,
            batch_size=parsed_arguments.batch_size,
            **dynamic_hyperparameters_dictionary
        )
        num_classes = len(train_dataset.classes)
        dataset_classes = train_dataset.classes

    if parsed_arguments.mode == "test":
        perform_sanity_check(
            train_dataset=train_dataset, num_classes=num_classes, base_learning_rate=parsed_arguments.learning_rate
        )

    elif parsed_arguments.mode == "optimize":
        run_optuna_study(train_dataset=train_dataset, validation_dataset=validation_dataset, num_classes=num_classes)

    elif parsed_arguments.mode == "train":
        execute_training_loop(
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            dataset_classes=dataset_classes,
            num_epochs=parsed_arguments.num_epochs,
            learning_rate=parsed_arguments.learning_rate,
            weight_decay=parsed_arguments.weight_decay,
            loss_type=parsed_arguments.loss_type,
            resume_flag=parsed_arguments.resume,
            resume_directory_path=parsed_arguments.resume_directory,
            **dynamic_hyperparameters_dictionary
        )

    elif parsed_arguments.mode == "infer":
        if not parsed_arguments.run_directory or not parsed_arguments.image_path:
            print("Error: Running inference requires specifying both --run_directory and --image_path")
            sys.exit(1)
        execute_inference(
            run_directory_path=parsed_arguments.run_directory, input_image_path=parsed_arguments.image_path
        )


if __name__ == "__main__":
    run_main_dispatcher()
