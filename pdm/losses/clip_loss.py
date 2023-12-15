import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, prompt_embeddings, arch_vectors):
        arch_vectors_similarity = F.softmax((arch_vectors @ arch_vectors.T) / self.temperature, dim=-1)
        texts_similarity = F.softmax((prompt_embeddings @ prompt_embeddings.T) / self.temperature, dim=-1)
        loss = F.cross_entropy(arch_vectors_similarity.T, texts_similarity.T, reduction='mean')
        return loss.mean()
