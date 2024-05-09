# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""
import os

from absl import app
from absl import flags
import distance
import embedding
import io_util
import numpy as np


_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)


def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


def save_ref_embeds(ref_dir, batch_size=32, max_count=-1):
    """Computes and saves the embeddings for the reference images.

    Args:
        ref_dir: Path to the directory containing reference images.
        batch_size: Batch size used in the CLIP embedding calculation.
        max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.
    """
    embedding_model = embedding.ClipEmbeddingModel()
    ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype("float32")

    # Save the embeddings to a file in the parent directory of the reference images.
    save_dir = os.path.dirname(ref_dir)
    name = os.path.basename(ref_dir)
    save_path = os.path.join(save_dir, f"cmmd_embs_{name}.npy")
    np.save(save_path, ref_embs)
