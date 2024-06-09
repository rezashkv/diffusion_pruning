from datasets import load_dataset
from pdm.datasets import load_cc3m_dataset, load_coco_dataset
import os
import random
import numpy as np
import requests
from torchvision import transforms
import torch
import PIL


def get_dataset(config):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    data_dir = getattr(config, "data_dir", None)

    train_data_dir = getattr(config, "train_data_dir", None)
    train_data_file = getattr(config, "train_data_file", None)

    validation_data_dir = getattr(config, "validation_data_dir", None)
    validation_data_file = getattr(config, "validation_data_file", None)

    if "conceptual_captions" in data_dir:
        dataset = {"train": load_cc3m_dataset(data_dir,
                                              split="train",
                                              split_file=train_data_file,
                                              split_dir=train_data_dir)}
        if validation_data_dir is not None:
            dataset["validation"] = load_cc3m_dataset(data_dir,
                                                      split="validation",
                                                      split_file=validation_data_file,
                                                      split_dir=validation_data_dir)

    elif "coco" in data_dir:
        year = config.year
        dataset = {"train": load_coco_dataset(os.path.join(data_dir, "images", f"train{year}"),
                                              os.path.join(data_dir, "annotations", f"captions_train{year}.json")),
                   "validation": load_coco_dataset(os.path.join(data_dir, "images", f"val{year}"),
                                                   os.path.join(data_dir, "annotations",
                                                                f"captions_val{year}.json"))}

    else:
        data_files = {}
        if config.data_dir is not None:
            data_files["train"] = os.path.join(config.data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=config.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    return dataset


def get_transforms(config):
    train_transform = transforms.Compose(
        [
            transforms.Resize(config.model.unet.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                config.model.unet.resolution) if config.data.dataloader.center_crop else transforms.RandomCrop(
                config.model.unet.resolution),
            transforms.RandomHorizontalFlip() if config.data.dataloader.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(config.model.unet.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                config.model.unet.resolution) if config.data.dataloader.center_crop else transforms.RandomCrop(
                config.model.unet.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return train_transform, validation_transform


def download_images_if_missing(samples, image_column):
    if isinstance(samples[image_column][0], str):
        if not os.path.exists(samples[image_column][0]):
            downloaded_images = []
            for image in samples[image_column]:
                try:
                    # download image and convert it to a PIL image
                    downloaded_images.append(PIL.Image.open(requests.get(image, stream=True).raw))
                except:
                    # remove the caption if the image is not found
                    downloaded_images.append(None)
            samples[image_column] = downloaded_images
        else:
            imgs = []
            for image in samples[image_column]:
                try:
                    imgs.append(PIL.Image.open(image))
                except:
                    samples[image_column] = None
                    continue
            samples[image_column] = imgs
    return samples


def tokenize_captions(samples, tokenizer, is_train=True):
    captions = []
    for caption in samples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column should contain either strings or lists of strings."
            )

    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def get_mpnet_embeddings(capts, mpnet_model, mpnet_tokenizer, is_train=True):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)

    captions = []
    for caption in capts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column  should contain either strings or lists of strings."
            )

    encoded_input = mpnet_tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        encoded_input = encoded_input.to(mpnet_model.device)
        model_output = mpnet_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def preprocess_samples(samples, tokenizer, mpnet_model, mpnet_tokenizer, transform, image_column="image",
                       caption_column="caption", is_train=True):
    samples = download_images_if_missing(samples, image_column)
    images = [image.convert("RGB") if image is not None else image for image in samples[image_column]]
    samples["pixel_values"] = [transform(image) if image is not None else image for image in images]
    samples["input_ids"] = tokenize_captions(samples[caption_column], tokenizer=tokenizer, is_train=is_train)
    samples["mpnet_embeddings"] = get_mpnet_embeddings(samples[caption_column], mpnet_model=mpnet_model,
                                                       mpnet_tokenizer=mpnet_tokenizer, is_train=is_train)
    return samples


def preprocess_prompts(samples, mpnet_model, mpnet_tokenizer):
    samples["mpnet_embeddings"] = get_mpnet_embeddings(samples["prompts"], mpnet_model=mpnet_model,
                                                       mpnet_tokenizer=mpnet_tokenizer, is_train=False)
    return samples


def collate_fn(samples):
    samples = [sample for sample in samples if sample["pixel_values"] is not None]
    if len(samples) == 0:
        return {"pixel_values": torch.tensor([]), "input_ids": torch.tensor([]),
                "mpnet_embeddings": torch.tensor([])}
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([sample["input_ids"] for sample in samples])
    mpnet_embeddings = torch.stack([sample["mpnet_embeddings"] for sample in samples])
    mpnet_embeddings = mpnet_embeddings.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, "input_ids": input_ids, "mpnet_embeddings": mpnet_embeddings}


def prompts_collator(samples):
    prompts = [sample["prompts"] for sample in samples]
    prompt_embdeddings = torch.stack([sample["mpnet_embeddings"] for sample in samples])
    prompt_embdeddings = prompt_embdeddings.to(memory_format=torch.contiguous_format).float()
    return {"prompts": prompts, "mpnet_embeddings": prompt_embdeddings}


def filter_dataset(dataset, hyper_net, quantizer, mpnet_model, mpnet_tokenizer, caption_column="caption"):
    train_captions = dataset["train"][caption_column]
    validation_captions = dataset["validation"][caption_column]
    train_filtering_dataloader = torch.utils.data.DataLoader(train_captions, batch_size=2048, shuffle=False)
    validation_filtering_dataloader = torch.utils.data.DataLoader(validation_captions, batch_size=2048,
                                                                  shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyper_net.to(device)
    quantizer.to(device)
    mpnet_model.to(device)
    hyper_net.eval()
    quantizer.eval()
    train_indices = []
    validation_indices = []
    with torch.no_grad():
        for batch in train_filtering_dataloader:
            batch = get_mpnet_embeddings(batch, mpnet_model, mpnet_tokenizer, is_train=True)
            arch_v = hyper_net(batch)
            indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
            train_indices.append(indices)
        for batch in validation_filtering_dataloader:
            batch = get_mpnet_embeddings(batch, mpnet_model, mpnet_tokenizer, is_train=False)
            arch_v = hyper_net(batch)
            indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
            validation_indices.append(indices)
    train_indices = torch.cat(train_indices, dim=0)
    validation_indices = torch.cat(validation_indices, dim=0)

    return train_indices, validation_indices
