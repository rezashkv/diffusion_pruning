import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, structure, temperature=2.0):
        super().__init__()
        self.temperature = temperature

        self.structure = structure
        self.width_list = [w for sub_width_list in self.structure['width'] for w in sub_width_list]
        self.width_list_sum = [sum(sub_width_list) for sub_width_list in self.structure['width']]
        self.depth_list = [d for sub_depth_list in self.structure['depth'] for d in sub_depth_list]

        width_indices = [0] + np.cumsum(self.width_list_sum).tolist()
        self.width_intervals = [(width_indices[i], width_indices[i + 1]) for i in range(len(width_indices) - 1)]

        widths_sum = sum(self.width_list) - 1

        self.depth_indices = (widths_sum + np.cumsum(self.depth_list)).tolist()

        template = torch.tensor(self.width_list + [d for d in self.depth_list if d != 0])
        #
        template = torch.repeat_interleave(template, template).type(torch.float32)
        self.template = (1.0 / template).requires_grad_(False)

    def forward(self, prompt_embeddings, arch_vectors):
        self.template = self.template.to(prompt_embeddings.device)

        # Multiply the slice of the arch_vectors defined by the start and end index of the width of the block with the
        # corresponding depth element of the arch_vectors.
        arch_vectors_clone = arch_vectors.clone()
        for i, elem in enumerate(self.depth_list):
            if elem != 0:
                arch_vectors_clone[:, self.width_intervals[i][0]:self.width_intervals[i][1]] = (
                        arch_vectors[:, self.width_intervals[i][0]:self.width_intervals[i][1]] *
                        arch_vectors[:, self.depth_indices[i]:(self.depth_indices[i]+1)])

        arch_vectors_ = arch_vectors_clone * torch.sqrt(self.template).detach()

        arch_vectors_similarity = F.softmax((arch_vectors_ @ arch_vectors_.T) / self.temperature, dim=-1)
        texts_similarity = F.softmax((prompt_embeddings @ prompt_embeddings.T) / self.temperature, dim=-1)
        loss = F.cross_entropy(arch_vectors_similarity.T, texts_similarity.T, reduction='mean')
        return loss.mean()
