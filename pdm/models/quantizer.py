# credit to taming-transformers:
# https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L112
from typing import Tuple

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config, ConfigMixin
from torch import nn
from projects.diffusion_pruning.pdm.utils.estimation_utils import gumbel_softmax_sample, hard_concrete, importance_gumbel_softmax_sample
from diffusers import ModelMixin
import torch.distributed as dist


class StructureVectorQuantizer(ModelMixin, ConfigMixin):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    @register_to_config
    def __init__(
            self,
            n_e: int,
            structure: dict,
            beta: float = 0.25,
            remap=None,
            unknown_index: str = "random",
            sane_index_shape: bool = True,
            temperature: float = 0.4,
            base: int = 2,
            depth_order: list = None,
            non_zero_width: bool = True,
            sinkhorn_epsilon: float = 0.05,
            sinkhorn_iterations: int = 3,
            resource_aware_normalization: bool = True,
            optimal_transport: bool = True,
    ):
        super().__init__()

        vq_embed_dim = 0
        for w_config, d_config in zip(structure['width'], structure['depth']):
            vq_embed_dim += sum(w_config)
            if d_config == [1]:
                # depth_indices.append(vq_embed_dim)
                vq_embed_dim += 1

        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta

        self.structure = structure

        self.width_list = [w for sub_width_list in self.structure['width'] for w in sub_width_list]
        self.width_list_sum = [sum(sub_width_list) for sub_width_list in self.structure['width']]
        width_indices = [0] + np.cumsum(self.width_list_sum).tolist()
        self.width_intervals = [(width_indices[i], width_indices[i + 1]) for i in range(len(width_indices) - 1)]

        self.depth_list = [d for sub_depth_list in self.structure['depth'] for d in sub_depth_list]
        widths_sum = sum(self.width_list) - 1
        self.depth_indices = (widths_sum + np.cumsum(self.depth_list)).tolist()

        num_depth_block = sum(self.depth_list)
        if depth_order is None:
            depth_order = list(range(num_depth_block))
        self.input_depth_order = depth_order
        self.depth_order = [i % num_depth_block for i in depth_order]

        template = torch.tensor(self.width_list + [d for d in self.depth_list if d != 0])
        template = torch.repeat_interleave(template, template).type(torch.float32)
        self.template = (1.0 / template).requires_grad_(False)
        self.prunable_macs_template = None
        if resource_aware_normalization is None:
            resource_aware_normalization = True
        self.resource_effect_normalization = resource_aware_normalization

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        nn.init.orthogonal_(self.embedding.weight)
        self.embedding_gs = nn.Parameter(self.embedding.weight.detach().clone(), requires_grad=False)

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
                f"Remapping {self.n_e} indices to {self.re_embed} indices."
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        self.temperature = temperature
        self.base = base

        # Avoid cases that width gets zero but we don't actually want to remove the whole block.
        self.non_zero_width = non_zero_width

        self.optimal_transport = optimal_transport
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

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

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        z = z.contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        if self.training:
            embedding_gs = self.gumbel_sigmoid_trick(self.embedding.weight)
            self.embedding_gs.data = embedding_gs.detach()
            if self.optimal_transport:
                min_encoding_indices = self.get_optimal_transport_min_encoding_indices(z_flattened)
            else:
                min_encoding_indices = self.get_cosine_sim_min_encoding_indices(z_flattened)
        else:
            embedding_gs = self.embedding_gs.detach()
            min_encoding_indices = self.get_cosine_sim_min_encoding_indices(z_flattened)

        z_q = embedding_gs[min_encoding_indices].view(z.shape)

        z_q_out = z_q.contiguous()

        perplexity = None
        min_encodings = None

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0])

        if not self.training:
            z_q_out = hard_concrete(z_q)

        return z_q_out, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...] = None) -> torch.Tensor:
        # shape specifying (batch, dim)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.Tensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.contiguous()

        return z_q

    def get_codebook_entry_gumbel_sigmoid(self, indices: torch.LongTensor, shape: Tuple[int, ...] = None,
                                          hard=False) -> torch.Tensor:
        z_q = self.get_codebook_entry(indices, shape).contiguous()
        if hard:
            return hard_concrete(self.gumbel_sigmoid_trick(z_q))
        else:
            return self.gumbel_sigmoid_trick(z_q)

    def gumbel_sigmoid_trick(self, z_q: torch.Tensor):
        num_width = sum(self.width_list)
        z_q_width = z_q[:, :num_width]
        z_q_depth = z_q[:, num_width:]

        fixed_seed = not self.training

        z_q_depth_b_ = importance_gumbel_softmax_sample(z_q_depth, temperature=self.temperature, offset=self.base,
                                                        fixed_seed=fixed_seed)
        z_q_depth_b = torch.zeros_like(z_q_depth_b_, device=z_q_depth_b_.device)
        z_q_depth_b[:, self.depth_order] = z_q_depth_b_

        z_q_width_list = self._transform_width_vector(z_q_width)
        z_q_width_b_list = [gumbel_softmax_sample(zw, temperature=self.temperature, offset=self.base,
                                                  force_width_non_zero=self.non_zero_width, fixed_seed=fixed_seed) for
                            zw in z_q_width_list]
        z_q_width_b = torch.cat(z_q_width_b_list, dim=1)

        z_q_out = torch.cat([z_q_width_b, z_q_depth_b], dim=1)
        return z_q_out

    def print_param_stats(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                print(f"{name}: {param.mean()}, {param.std()}")

    def _transform_width_vector(self, inputs):
        assert inputs.shape[1] == sum(self.width_list)
        arch_vector = []
        start = 0
        for i in range(len(self.width_list)):
            end = start + self.width_list[i]
            arch_vector.append(inputs[:, start:end])
            start = end

        return arch_vector

    def width_depth_normalize(self, inputs):
        self.template = self.template.to(inputs.device)
        if self.resource_effect_normalization:
            self.prunable_macs_template = self.prunable_macs_template.to(inputs.device)
        # Multiply the slice of the arch_vectors defined by the start and end index of the width of the block with the
        # corresponding depth element of the arch_vectors.
        inputs_clone = hard_concrete(inputs.clone())
        for i, elem in enumerate(self.depth_list):
            if elem != 0:
                inputs_clone[:, self.width_intervals[i][0]:self.width_intervals[i][1]] = (
                        inputs[:, self.width_intervals[i][0]:self.width_intervals[i][1]] *
                        inputs[:, self.depth_indices[i]:(self.depth_indices[i] + 1)])

        outputs = inputs_clone * (torch.sqrt(self.template).detach())
        if self.resource_effect_normalization:
            outputs = outputs * self.prunable_macs_template.detach()

        return outputs

    def set_prunable_macs_template(self, prunable_macs_list):
        depth_template = []
        for i, elem in enumerate(self.depth_list):
            if elem == 1:
                depth_template.append([sum(prunable_macs_list[i])])
        prunable_macs_list += depth_template
        prunable_macs_list = [item for sublist in prunable_macs_list for item in sublist]
        self.prunable_macs_template = torch.repeat_interleave(torch.tensor(prunable_macs_list),
                                                               torch.tensor(self.width_list + [1 for _ in range(
                                                                   len(depth_template))]))

    @torch.no_grad()
    def get_cosine_sim_min_encoding_indices(self, z: torch.Tensor) -> torch.Tensor:
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        u = self.width_depth_normalize(self.gumbel_sigmoid_trick(z))
        u = u / u.norm(dim=-1, keepdim=True)
        v = self.width_depth_normalize(self.embedding_gs)
        v = v / v.norm(dim=-1, keepdim=True)
        min_encoding_indices = torch.argmax(u @ v.t(), dim=-1)
        return min_encoding_indices

    @torch.no_grad()
    def get_optimal_transport_min_encoding_indices(self, a: torch.Tensor) -> torch.Tensor:
        # credit to
        # https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/main_swav.py#L354
        @torch.no_grad()
        def distributed_sinkhorn(out):
            Q = torch.exp(out / self.sinkhorn_epsilon).t()  # Q is K-by-B for consistency with notations from the paper
            B = Q.shape[1] * dist.get_world_size()
            K = Q.shape[0]

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            for it in range(self.sinkhorn_iterations):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                dist.all_reduce(sum_of_rows)
                Q /= sum_of_rows
                Q /= K

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B  # the columns must sum to 1 so that Q is an assignment
            return Q.t()

        @torch.no_grad()
        def sinkhorn(out):
            Q = torch.exp(out / self.sinkhorn_epsilon).t()  # Q is K-by-B for consistency with notations from the paper
            B = Q.shape[1]
            K = Q.shape[0]

            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            for it in range(self.sinkhorn_iterations):
                # normalize each row: total weight per prototype must be 1/K
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q /= K

                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B  # the colomns must sum to 1 so that Q is an assignment
            return Q.t()

        a = self.gumbel_sigmoid_trick(a)
        a = self.width_depth_normalize(a)
        a = a / a.norm(dim=-1, keepdim=True)

        codes = self.embedding_gs
        codes = self.width_depth_normalize(codes)
        codes = codes / codes.norm(dim=-1, keepdim=True)

        out = a @ codes.t()

        if dist.is_initialized():
            Q = distributed_sinkhorn(out)
        else:
            Q = sinkhorn(out)
        min_encoding_indices = torch.argmax(Q, dim=-1)
        return min_encoding_indices
