import os
import shutil

if __name__ == "__main__":

    base_path = "dataset/plantnet_300K/images"
    discard_path = "dataset/plantnet_300K/discarded"
    splits = ["train", "val", "test"]

    for split in splits:
        os.makedirs(os.path.join(discard_path, split), exist_ok=True)

    min_images = 64
    classes_before = len(os.listdir(os.path.join(base_path, "train")))
    classes_discarded = 0

    for class_folder in os.listdir(os.path.join(base_path, "train")):
        train_path = os.path.join(base_path, "train", class_folder)

        if not os.path.isdir(train_path):
            continue

        num_images = len(os.listdir(train_path))

        if num_images < min_images:
            classes_discarded += 1
            for split in splits:
                source = os.path.join(base_path, split, class_folder)
                dest = os.path.join(discard_path, split, class_folder)
                if os.path.exists(source):
                    shutil.move(source, dest)
