import torch
from torch.utils.data import Sampler


class EqualBatchSampler(Sampler):
    def __init__(self, patches: list, batch_size: int, epoch_length: int | None = None, generator=None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("balanced batch_size must be an even integer of at least two")
        self.batch_size = batch_size
        self.generator = generator or torch.Generator()

        self.pos_indices = [i for i, patch in enumerate(patches) if patch["has_microbleed"]]
        self.neg_indices = [i for i, patch in enumerate(patches) if not patch["has_microbleed"]]

        if not self.pos_indices or not self.neg_indices:
            raise ValueError("balanced sampling requires positive and negative patches")
        self.n_pos = self.batch_size // 2
        self.n_neg = self.batch_size // 2
        self.num_batches = epoch_length or max(
            (len(self.pos_indices) + self.n_pos - 1) // self.n_pos,
            (len(self.neg_indices) + self.n_neg - 1) // self.n_neg,
        )

    def __iter__(self):
        for _ in range(self.num_batches):
            positive = torch.randint(len(self.pos_indices), (self.n_pos,), generator=self.generator)
            negative = torch.randint(len(self.neg_indices), (self.n_neg,), generator=self.generator)
            batch = [self.pos_indices[index] for index in positive.tolist()]
            batch.extend(self.neg_indices[index] for index in negative.tolist())
            order = torch.randperm(self.batch_size, generator=self.generator).tolist()
            yield [batch[index] for index in order]

    def __len__(self):
        return self.num_batches
