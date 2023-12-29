import os

def load_dataset_dir(dataset_dir):
    dataset = []
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    for image_file in image_files:
        image_path = os.path.join(dataset_dir, image_file)
        caption_file = image_file.replace('.jpg', '.txt')
        caption_path = os.path.join(dataset_dir, caption_file)
        with open(caption_path, 'r') as caption_file:
            caption = caption_file.read()
        example = {
            'image': image_path,
            'caption': str(caption),
        }
        dataset.append(example)
    return dataset


def load_main_laion_dataset(main_dataset_dir, train_dirs):
    train_datasets = {}
    for subdir in train_dirs:
        dataset_name = subdir
        dataset_dir = os.path.join(main_dataset_dir, subdir)
        dataset = load_dataset_dir(dataset_dir)
        train_datasets[dataset_name] = dataset

    return train_datasets

