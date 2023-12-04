import os
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
from torch import nn
from diffusion_pruning.models.utils import gumbel_softmax_sample, hard_concrete
from diffusers import ModelMixin


class StructureVectorQuantizer(ModelMixin):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
            self,
            n_e: int,
            vq_embed_dim: int = 256,
            beta: float = 0.25,
            remap=None,
            unknown_index: str = "random",
            sane_index_shape: bool = False,
            temperature: float = 0.4,
            base: int = 2,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        self.temperature = temperature
        self.base = base

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        # reshape z -> (batch, dim) and flatten
        z = z.contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q: torch.FloatTensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0])

        z_q = gumbel_softmax_sample(z_q, temperature=self.temperature, offset=self.base)
        if not self.training:
            z_q = hard_concrete(z_q)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.FloatTensor:
        # shape specifying (batch, dim)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.FloatTensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.contiguous()

        return z_q

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            save_function: Callable = None,
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
    ):
        # save the hyper_net state
        torch.save(self.state_dict(), os.path.join(save_directory, "quantizer.pt"))

    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # load the hyper_net state
        model = cls(**kwargs)
        model.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, "quantizer.pt")))
        return model

    def print_param_stats(self):
        print("Quantizer")
        for name, param in self.named_parameters():
            if "weight" in name:
                print(f"{name}: {param.mean()}, {param.std()}")
        print("")
