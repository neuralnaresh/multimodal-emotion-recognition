import typing
import itertools

import torch

from losses.nce import NCELoss

from losses.config.avid import AVIDMemoryConfig, AVIDLossConfig

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def to(self, device):
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

class AVIDMemory(torch.nn.Module):
    def __init__(self, samples: int, utterances: int, embedding_sizes: typing.Dict[str, int], config: AVIDMemoryConfig) -> None:
        super().__init__()

        self.utterances = utterances

        self.negatives = config.negatives
        self.momentum = [config.momentum] * len(embedding_sizes)
        self.temperature = config.temperature

        self.inter = config.inter
        self.intra = config.intra

        self.multinomial = AliasMethod(torch.ones(samples - 1))

        self._init_memory(samples, utterances, embedding_sizes)

    def _init_memory(self, samples: int, utterances: int, embedding_sizes: typing.Dict[str, int]) -> None:
        for modality, size in embedding_sizes.items():
            self.register_buffer(f'{modality}_memory', torch.nn.functional.normalize(torch.randn(samples * utterances, size), p=2, dim=1))

    def _update_memory(self, embeddings: typing.Dict[str, torch.Tensor], indices: torch.Tensor) -> None:
        momentums = {modality: float(momentum) for modality, momentum in zip(embeddings.keys(), self.momentum)}

        with torch.no_grad():
            for modality, embeddings in embeddings.items():
                positives = getattr(self, f'{modality}_memory').index_select(0, indices.view(-1))

                positives = positives.mul(momentums[modality])
                positives = positives.add(embeddings.mul(1 - momentums[modality]).squeeze(dim=-1))

                positives = torch.nn.functional.normalize(positives, p=2, dim=1)

                getattr(self, f'{modality}_memory').index_copy_(0, indices.view(-1), positives)
        
    def _sample_negatives(self, indices: torch.Tensor, utterance_lengths: typing.List[int], negatives: int):
        utterances = sum(utterance_lengths)  if utterance_lengths is not None else indices.shape[0]

        indices = self._convert_indices(indices, utterance_lengths)

        samples = self.multinomial.draw(utterances * negatives).view(utterances, -1).to(indices.device)
        samples = samples + (samples >= indices.unsqueeze(dim=1)).long()

        return samples

    def _convert_indices(self, indices: torch.Tensor, utterance_lengths: typing.List[int]) -> torch.Tensor:
        if utterance_lengths is None:
            return indices

        result = []

        for i, length in enumerate(utterance_lengths):
            result.extend(indices[i, :length].tolist())
        
        return torch.tensor(result, device=indices.device, dtype=torch.long)

    def _compute_scores(self, context: torch.Tensor, targets: typing.List[torch.Tensor], temperature: float) -> torch.Tensor:
        return [torch.bmm(target, context).squeeze(dim=-1) / temperature for target in targets]

    def forward(self, embeddings: typing.Dict[str, torch.Tensor], indices: torch.Tensor, utterance_lengths: typing.List[int]) -> typing.Dict[str, torch.Tensor]:
        utterances = sum(utterance_lengths) if utterance_lengths is not None else indices.shape[0]

        normalized = {modality: torch.nn.functional.normalize(embeddings, p=2, dim=1) for modality, embeddings in embeddings.items()}

        for modality in normalized:
            result = []
            
            if utterance_lengths is not None:
                for i, length in enumerate(utterance_lengths):
                    result.append(normalized[modality][i, :length])
            else:
                result.append(normalized[modality])
            
            normalized[modality] = torch.cat(result, dim=0).unsqueeze(dim=-1)

        with torch.no_grad():
            positives = {modality: getattr(self, f'{modality}_memory')[self._convert_indices(indices, utterance_lengths)].view(utterances, 1, -1) for modality in normalized.keys()}
            negatives = {modality: getattr(self, f'{modality}_memory')[self._sample_negatives(indices, utterance_lengths, self.negatives)].view(utterances, self.negatives, -1) for modality in normalized.keys()}

        scores = {}

        if self.inter:
            for a, b in [(a, b) for a, b in list(itertools.product(embeddings.keys(), repeat=2)) if a != b]:
                scores[f'(inter){a}_{b}'] = self._compute_scores(normalized[a], [positives[b], negatives[b]], self.temperature)
                    
        if self.intra:
            for modality in embeddings:
                scores[f'(intra){modality}'] = self._compute_scores(normalized[modality], [positives[modality], negatives[modality]], self.temperature)

        self._update_memory(normalized, self._convert_indices(indices, utterance_lengths))

        return scores

class AVIDLoss(torch.nn.Module):
    def __init__(self, samples: int, utterances: int, embedding_sizes: typing.Dict[str, int], config: AVIDLossConfig) -> None:
        super().__init__()

        coefficients = config.inter + config.intra

        self.inter = config.inter / coefficients
        self.intra = config.intra / coefficients

        self.memory = AVIDMemory(samples, utterances, embedding_sizes, config.memory)
        self.nce = NCELoss()

    def forward(self, embeddings: typing.Dict[str, torch.Tensor], indices: torch.Tensor, _, utterance_lengths: torch.Tensor) -> torch.Tensor:
        scores = self.memory(embeddings, indices, utterance_lengths)

        inter_modal_loss = 0
        intra_modal_loss = 0

        for score in scores:
            loss = self.nce(*scores[score]) / float(len(embeddings))
            
            if '(inter)' in score:
                inter_modal_loss += loss
            elif '(intra)' in score:
                intra_modal_loss += loss

        return self.inter * inter_modal_loss + self.intra * intra_modal_loss