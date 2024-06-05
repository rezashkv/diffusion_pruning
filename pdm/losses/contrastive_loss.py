import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, arch_vector_temperature=1.0, prompt_embedding_temperature=1.0):
        super().__init__()
        self.arch_vector_temperature = arch_vector_temperature
        self.prompt_embedding_temperature = prompt_embedding_temperature

    def forward(self, prompt_embeddings, arch_vectors, return_similarity=False):
        arch_vectors_normalized = arch_vectors / arch_vectors.norm(dim=1, keepdim=True)
        prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(dim=1, keepdim=True)
        arch_vectors_similarity = F.softmax(
            (arch_vectors_normalized @ arch_vectors_normalized.T) / self.arch_vector_temperature, dim=-1)
        texts_similarity = F.softmax((prompt_embeddings @ prompt_embeddings.T) / self.prompt_embedding_temperature,
                                     dim=-1)
        loss = F.binary_cross_entropy(arch_vectors_similarity.T, texts_similarity.T, reduction='mean')
        if return_similarity:
            return loss, arch_vectors_similarity.detach().cpu().numpy()
        else:
            return loss
