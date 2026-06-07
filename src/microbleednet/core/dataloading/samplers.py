import random
from torch.utils.data import Sampler


class EqualBatchSampler(Sampler):
    def __init__(self, patches: list, batch_size: int):
        self.batch_size = batch_size

        self.pos_indices = [i for i, patch in enumerate(patches) if patch.get("has_microbleed")]
        self.neg_indices = [i for i, patch in enumerate(patches) if not patch.get("has_microbleed")]

        print(f"Total Patches: {len(patches)}, Positive: {len(self.pos_indices)}, Negative: {len(self.neg_indices)}")

        self.n_pos = self.batch_size // 2
        self.n_neg = self.batch_size // 2
        self.num_batches = len(self.pos_indices) // self.n_pos

    def __iter__(self):
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)

        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.num_batches):
            batch = []

            # Fetch Positive Indices
            for _ in range(self.n_pos):
                if pos_ptr >= len(self.pos_indices):
                    random.shuffle(self.pos_indices)
                    pos_ptr = 0
                batch.append(self.pos_indices[pos_ptr])
                pos_ptr += 1

            # Fetch Negative Indices
            for _ in range(self.n_neg):
                if neg_ptr >= len(self.neg_indices):
                    random.shuffle(self.neg_indices)
                    neg_ptr = 0
                batch.append(self.neg_indices[neg_ptr])
                neg_ptr += 1

            # Shuffle the final batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
