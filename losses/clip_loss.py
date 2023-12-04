import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, prompt_embeddings, arch_vectors):
        # logits = (prompt_embeddings @ arch_vectors.T) / self.temperature
        arch_vectors_similarity = F.softmax(arch_vectors @ arch_vectors.T, dim=-1)
        texts_similarity = F.softmax(prompt_embeddings @ prompt_embeddings.T, dim=-1)
        loss = F.cross_entropy(arch_vectors_similarity.T, texts_similarity.T, reduction='mean')
        # targets = F.softmax(
        #     (arch_vectors_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        # )
        # texts_loss = F.cross_entropy(logits, targets, reduction='none')
        # arch_vectors_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        # loss = (arch_vectors_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()