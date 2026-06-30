import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image


def verify_dataset_integrity(train_directory, validation_directory):
    is_valid_flag = True
    for dataset_split_name, directory_path in [("train", train_directory), ("val", validation_directory)]:
        if not os.path.exists(directory_path):
            print(f"Error: {dataset_split_name} directory not found at: {directory_path}")
            is_valid_flag = False
            continue

        class_directories = sorted(
            [
                category_directory
                for category_directory in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, category_directory))
            ]
        )

        if not class_directories:
            print(f"Error: No class subdirectories found in: {directory_path}")
            is_valid_flag = False

        for category in class_directories:
            category_full_path = os.path.join(directory_path, category)
            image_file_names = os.listdir(category_full_path)
            if not image_file_names:
                print(f"Warning: Category directory is empty: {category_full_path}")
                is_valid_flag = False

            for file_name in image_file_names:
                file_full_path = os.path.join(category_full_path, file_name)
                try:
                    with Image.open(file_full_path) as opened_image:
                        opened_image.verify()
                except Exception as exception_info:
                    print(f"Corrupted image detected at: {file_full_path}. Reason: {exception_info}")
                    is_valid_flag = False

    if is_valid_flag:
        print("Dataset alignment and integrity successfully verified!")
    else:
        print("Dataset integrity checks failed. Review warnings above.")


def generate_history_plots(run_directory_path):
    history_file_path = os.path.join(run_directory_path, "model_history.json")
    if not os.path.exists(history_file_path):
        print(f"Error: No history file found at: {history_file_path}")
        return

    with open(history_file_path, "r") as json_file_pointer:
        history_tracker = json.load(json_file_pointer)

    epochs_count = len(history_tracker["train"]["loss"])
    epochs_range = list(range(1, epochs_count + 1))

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history_tracker["train"]["loss"], label="Train Loss")
    axes[0].plot(epochs_range, history_tracker["validation"]["loss"], label="Val Loss")
    axes[0].set_title("Loss History")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history_tracker["train"]["accuracy_top1"], label="Train Top-1")
    axes[1].plot(epochs_range, history_tracker["validation"]["accuracy_top1"], label="Val Top-1")
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Percentage")
    axes[1].legend()

    axes[2].plot(epochs_range, history_tracker["train"]["f1_score"], label="Train F1")
    axes[2].plot(epochs_range, history_tracker["validation"]["f1_score"], label="Val F1")
    axes[2].set_title("Macro F1-Score")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Percentage")
    axes[2].legend()

    plt.tight_layout()
    output_plot_path = os.path.join(run_directory_path, "training_metrics_plots.png")
    plt.savefig(output_plot_path)
    print(f"Plots successfully exported to: {output_plot_path}")


def run_tools_dispatcher():
    argument_parser = argparse.ArgumentParser(description="Flora Utilities Tool")
    argument_parser.add_argument("--mode", type=str, required=True, choices=["verify", "plot", "download"])
    argument_parser.add_argument("--train_directory", type=str, default="dataset/plantnet_300K/images/train")
    argument_parser.add_argument("--validation_directory", type=str, default="dataset/plantnet_300K/images/val")
    argument_parser.add_argument("--run_directory", type=str, default=None)

    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.mode == "verify":
        verify_dataset_integrity(
            train_directory=parsed_arguments.train_directory, validation_directory=parsed_arguments.validation_directory
        )
    elif parsed_arguments.mode == "plot":
        if not parsed_arguments.run_directory:
            print("Error: Plot mode requires specifying --run_directory")
            sys.exit(1)
        generate_history_plots(run_directory_path=parsed_arguments.run_directory)
    elif parsed_arguments.mode == "download":
        print("Download automation not implemented under strict YAGNI constraints. Please source dataset manually.")


if __name__ == "__main__":
    run_tools_dispatcher()
