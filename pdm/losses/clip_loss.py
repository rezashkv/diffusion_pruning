import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, structure, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.structure = structure

        template = []
        depth_indices = []
        depth_corresponding_width_indices = []
        dim = 0

        for elem in self.structure:
            if "width" in elem:
                elem["width"] = sum(elem["width"])
                dim += elem["width"]
                template.append(elem["width"])
            if "depth" in elem:
                elem["depth"] = sum(elem["depth"])
                template.append(elem["depth"])
                depth_indices.append(dim)
                depth_corresponding_width_indices.append((dim - elem["width"], dim))
                dim += elem["depth"]

        self.depth_indices = depth_indices
        self.depth_corresponding_width_indices = depth_corresponding_width_indices
        self.width_indices_tensor = torch.tensor([i for i in range(dim) if i not in depth_indices])

        self.template = 1.0 / torch.tensor([elem for elem in template for _ in range(elem)],
                                           dtype=torch.float32).requires_grad_(False)

    def forward(self, prompt_embeddings, arch_vectors):
        self.template = self.template.to(prompt_embeddings.device)
        # corresponding_width_indices is a list of tuples, each tuple contains the start and end index of a layer.
        # multiply the slice of the arch_vectors defined by the start and end index of a layer with the corresponding
        # depth element of the arch_vectors.
        arch_vectors_clone = arch_vectors.clone()
        for i, (start, end) in enumerate(self.depth_corresponding_width_indices):
            arch_vectors_clone[:, start:end] = (arch_vectors[:, start:end] * arch_vectors[:, self.depth_indices[i]:self.depth_indices[i] + 1])

        # multiply the arch_vectors with the template
        arch_vectors_ = arch_vectors_clone * torch.sqrt(self.template).detach()

        arch_vectors_similarity = F.softmax((arch_vectors_ @ arch_vectors_.T) / self.temperature, dim=-1)
        texts_similarity = F.softmax((prompt_embeddings @ prompt_embeddings.T) / self.temperature, dim=-1)
        loss = F.cross_entropy(arch_vectors_similarity.T, texts_similarity.T, reduction='mean')
        return loss.mean()