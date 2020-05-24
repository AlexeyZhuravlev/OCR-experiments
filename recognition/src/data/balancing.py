import numpy as np
from typing import List, Iterator, Tuple
from torch.utils.data import Sampler, Dataset, ConcatDataset

def build_balanced_concatenation(
        datasets_with_weights: List[Tuple[Dataset, int]],
        mini_epoch_len: int) -> Tuple[Dataset, Sampler]:
    """
    Gets list of datasets with associated weights, returns dataset concatenation
    and sampler object, which forms mini-epochs of total size mini_epoch_len
    using each source to get elements count proportional to corresponding weight

    I.e. one might want to use balanced concatenation of synthetic dataset with size 20M
    and real data with size 50K to get mini-epochs of size 200K for dataloader.
    When he passes [(synth_dataset, 1), (real_dataset, 1)], 200.000
    and obtains mini-epochs of 100.000 random synthetic elements and 100.000 random real elements
    On next iteration new synthetic elements will be obtained (from cyclic buffer),
    while real samples will be re-sampled from full set with returns
    """
    datasets = []
    mixture_infos = []
    for dataset, weight in datasets_with_weights:
        datasets.append(dataset)
        mixture_infos.append(MixtureSourceInfo(
            source_len=len(dataset),
            source_weight=weight
        ))
    concat_dataset = ConcatDataset(datasets)
    concat_sampler = SourceBalancedMiniEpochSampler(
        sources=mixture_infos,
        mini_epoch_len=mini_epoch_len)

    return concat_dataset, concat_sampler

class OffsetMiniEpochIndicesProvider:
    """
    Provides mini-epochs of indices like MiniEpochSampler, but adds specified offset to them
    Drops last batch for large datasets and performs per_epoch shuffling
    """
    def __init__(self, data_len, mini_epoch_len, offset):

        self.data_len = int(data_len)
        self.mini_epoch_len = int(mini_epoch_len)

        self.steps = int(data_len / self.mini_epoch_len)
        self.state_i = 0

        if self.steps == 0:
            self.divider = 1
        else:
            self.divider = self.steps

        self._indices = np.arange(self.data_len) + offset
        self.indices = self._indices
        self.end_pointer = max(self.data_len, self.mini_epoch_len)

    def shuffle(self) -> None:
        """Shuffles dataset per epoch"""
        if self.state_i == 0:
            if self.data_len >= self.mini_epoch_len:
                self.indices = self._indices
                np.random.shuffle(self.indices)
            else:
                self.indices = np.random.choice(
                    self._indices, self.mini_epoch_len, replace=True
                )

    def get_next_indices(self) -> np.array:
        """Returns next mini epoch of indices"""
        self.state_i = self.state_i % self.divider
        self.shuffle()

        start = self.state_i * self.mini_epoch_len
        stop = (self.state_i + 1) * self.mini_epoch_len
        indices = self.indices[start:stop]

        self.state_i += 1
        return indices

class MixtureSourceInfo:
    """
    Information about source in datasets mixture
    source_len: number of elements in the source
    source_weight: weight of this source in total mixture (might be unnormalized))
    """
    def __init__(self, source_len: int, source_weight: float):
        assert source_weight >= 0
        assert source_len > 0

        self.source_len = source_len
        self.source_weight = source_weight

class SourceBalancedMiniEpochSampler(Sampler):
    """
    Sampler class, to work on top of Concat dataset. Takes list of sources for concatenation
    with information about each length and each weight in the mixture.
    Samples mini-epochs of length mini_epoch_len, where amount of element from each source
    is proportional to specified weight
    """
    def __init__(self, sources: List[MixtureSourceInfo], mini_epoch_len: int):
        super().__init__(self)

        self.mini_epoch_len = mini_epoch_len
        self.providers = []

        total_weight = 0

        for source in sources:
            total_weight += source.source_weight

        current_length = 0
        current_offset = 0
        for i, source in enumerate(sources):
            if i == len(sources) - 1:
                source_elements = mini_epoch_len - current_length
            else:
                source_elements = round(
                    float(source.source_weight) * mini_epoch_len / total_weight
                )
            self.providers.append(
                OffsetMiniEpochIndicesProvider(
                    source.source_len,
                    source_elements,
                    current_offset
                )
            )
            current_length += source_elements
            current_offset += source.source_len

    def __iter__(self) -> Iterator[int]:
        all_indices = []
        for provider in self.providers:
            all_indices.append(provider.get_next_indices())

        indices_concat = np.concatenate(all_indices)
        np.random.shuffle(indices_concat)
        indices = indices_concat.tolist()

        return iter(indices)

    def __len__(self) -> int:
        return self.mini_epoch_len
