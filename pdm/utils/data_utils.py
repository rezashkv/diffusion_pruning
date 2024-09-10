from datasets import load_dataset
from ..datasets import load_cc3m_dataset, load_coco_dataset
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

    if data_dir is None:
        data_dir = ""

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
        if year == "2014_30k":
            train_year = "2014"
        else:
            train_year = year
        dataset = {"train": load_coco_dataset(os.path.join(data_dir, "images", f"train{train_year}"),
                                              os.path.join(data_dir, "annotations",
                                                           f"captions_train{train_year}.json")),
                   "validation": load_coco_dataset(os.path.join(data_dir, "images", f"val{year}"),
                                                   os.path.join(data_dir, "annotations",
                                                                f"captions_val{year}.json"))}

    else:
        assert config.dataset_name is not None, "Please provide a dataset name."
        dataset = load_dataset(config.dataset_name)
        if "validation" not in dataset:
            dataset_ = dataset["train"].train_test_split(test_size=0.001, seed=42)
            dataset["train"] = dataset_["train"]
            dataset["validation"] = dataset_["test"]
            del dataset_

    return dataset


def get_transforms(config):
    train_transform = transforms.Compose(
        [
            transforms.Resize(config.model.prediction_model.resolution,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                config.model.prediction_model.resolution) if config.data.dataloader.center_crop else transforms.RandomCrop(
                config.model.prediction_model.resolution),
            transforms.RandomHorizontalFlip() if config.data.dataloader.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(config.model.prediction_model.resolution,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                config.model.prediction_model.resolution) if config.data.dataloader.center_crop else transforms.RandomCrop(
                config.model.prediction_model.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return train_transform, validation_transform


def download_images_if_missing(samples):
    if isinstance(samples[0], str):
        if not os.path.exists(samples[0]):
            downloaded_images = []
            for image in samples:
                try:
                    # download image and convert it to a PIL image
                    downloaded_images.append(PIL.Image.open(requests.get(image, stream=True).raw))
                except:
                    # remove the caption if the image is not found
                    downloaded_images.append(None)
            samples = downloaded_images
        else:
            imgs = []
            for image in samples:
                try:
                    imgs.append(PIL.Image.open(image))
                except:
                    samples = None
                    continue
            samples = imgs
    return samples


def maybe_keep_random_caption(samples, is_train=True):
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
    return captions


def tokenize_caption(caption, tokenizer, max_length=None):
    if max_length is None:
        max_length = tokenizer.model_max_length

    inputs = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
    )
    return inputs.input_ids


def encode_prompt(
        tokenizer,
        text_encoder,
        prompt: str,
        max_sequence_length=77,
        device=None,
        text_input_ids=None,
        pooled=False,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    if not pooled:
        text_encoder = text_encoder.to(device)
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    else:
        text_encoder = text_encoder.to(device)
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
        # Use pooled output
        prompt_embeds = prompt_embeds.pooler_output

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds


def encode_prompt_with_multiple_encoders(
        tokenizers,
        text_encoders,
        prompt: str,
        max_sequence_length,
        device=None,
        text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = encode_prompt(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        max_sequence_length=max_sequence_length[0],
        prompt=prompt,
        device=device,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
        pooled=True,
    )

    prompt_embeds = encode_prompt(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length[1],
        prompt=prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
        pooled=False,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def encode_with_mpnet(caption, mpnet_model, mpnet_tokenizer):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)

    encoded_input = mpnet_tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        encoded_input = encoded_input.to(mpnet_model.device)
        model_output = mpnet_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def preprocess_sample(samples, tokenizers, text_encoders, mpnet_model, mpnet_tokenizer, transform, image_column="image",
                      caption_column="caption", is_train=True, max_sequence_length=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples[image_column] = download_images_if_missing(samples[image_column])
    images = [img.convert("RGB") if img is not None else img for
              img in samples[image_column]]
    samples["pixel_values"] = [transform(image) if image is not None else image for image in images]
    samples["mpnet_embeddings"] = encode_with_mpnet(samples[caption_column], mpnet_model=mpnet_model,
                                                    mpnet_tokenizer=mpnet_tokenizer)
    samples[caption_column] = maybe_keep_random_caption(samples[caption_column], is_train)
    if len(tokenizers) > 1:
        assert len(tokenizers) == len(max_sequence_length), "Number of tokenizers should be equal to the number of " \
                                                            "max_sequence_length values."
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_with_multiple_encoders(
            tokenizers, text_encoders, samples[caption_column], max_sequence_length, device=device)
        samples["prompt_embeds"] = prompt_embeds
        samples["pooled_prompt_embeds"] = pooled_prompt_embeds
        samples["text_ids"] = text_ids
    else:
        samples["prompt_embeds"] = encode_prompt(tokenizers[0], text_encoders[0], samples[caption_column],
                                                 max_sequence_length, device=device)

    return samples


def preprocess_prompts(samples, mpnet_model, mpnet_tokenizer):
    samples["prompts"] = maybe_keep_random_caption(samples["prompts"], is_train=False)
    samples["mpnet_embeddings"] = encode_with_mpnet(samples["prompts"], mpnet_model=mpnet_model,
                                                    mpnet_tokenizer=mpnet_tokenizer)
    return samples


def collate_fn(samples):
    samples = [sample for sample in samples if sample["pixel_values"] is not None]
    if len(samples) == 0:
        return {"pixel_values": torch.tensor([]), "prompt_embeds": torch.tensor([]),
                "mpnet_embeddings": torch.tensor([])}

    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_embeds = torch.stack([sample["prompt_embeds"] for sample in samples])
    prompt_embeds = prompt_embeds.to(memory_format=torch.contiguous_format).float()
    mpnet_embeddings = torch.stack([sample["mpnet_embeddings"] for sample in samples])
    mpnet_embeddings = mpnet_embeddings.to(memory_format=torch.contiguous_format).float()
    return_dict = {"pixel_values": pixel_values, "prompt_embeds": prompt_embeds, "mpnet_embeddings": mpnet_embeddings}

    if "text_ids" in samples[0]:
        text_ids = torch.stack([sample["text_ids"] for sample in samples])
        text_ids = text_ids.to(memory_format=torch.contiguous_format).long()
        return_dict["text_ids"] = text_ids
    if "pooled_prompt_embeds" in samples[0]:
        pooled_prompt_embeds = torch.stack([sample["pooled_prompt_embeds"] for sample in samples])
        pooled_prompt_embeds = pooled_prompt_embeds.to(memory_format=torch.contiguous_format).float()
        return_dict["pooled_prompt_embeds"] = pooled_prompt_embeds
    return return_dict


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
            batch = encode_with_mpnet(batch, mpnet_model, mpnet_tokenizer, is_train=True)
            arch_v = hyper_net(batch)
            indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
            train_indices.append(indices)
        for batch in validation_filtering_dataloader:
            batch = encode_with_mpnet(batch, mpnet_model, mpnet_tokenizer, is_train=False)
            arch_v = hyper_net(batch)
            indices = quantizer.get_cosine_sim_min_encoding_indices(arch_v)
            validation_indices.append(indices)
    train_indices = torch.cat(train_indices, dim=0)
    validation_indices = torch.cat(validation_indices, dim=0)

    return train_indices, validation_indices
