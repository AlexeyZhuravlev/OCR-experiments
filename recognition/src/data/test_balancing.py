from torch.utils.data import Dataset, DataLoader
from .balancing import build_balanced_concatenation
from collections import Counter

class DummyDataset(Dataset):
    def __init__(self, key, length):
        self.length = length
        self.key = key

    def __getitem__(self, idx):
        item = {
            "key": self.key,
            "idx": idx
        }
        return item

    def __len__(self):
        return self.length

def test_balanced():
    first_dataset = DummyDataset("A", 25)
    second_dataset = DummyDataset("B", 50)
    third_dataset = DummyDataset("C", 6)
    datasets_with_weights = [
        (first_dataset, 1),
        (second_dataset, 1),
        (third_dataset, 2)
    ]
    mini_epoch_len = 20

    concat, sampler = build_balanced_concatenation(datasets_with_weights, mini_epoch_len)

    loader = DataLoader(concat, batch_size=1, sampler=sampler, shuffle=False)
    num_epochs = 10

    key_to_elems = {"A": [], "B": [], "C": []}
    for _ in range(num_epochs):
        values = []
        key_counters = {"A": 0, "B": 0, "C": 0}
        for item in loader:
            key = item["key"][0]
            value = item["idx"][0].item()
            values.append((key, value))
            key_counters[key] += 1
            key_to_elems[key].append(value)
        assert key_counters["A"] == 5
        assert key_counters["B"] == 5
        assert key_counters["C"] == 10

        print(values)

    a_counter = Counter(key_to_elems["A"])
    assert sorted(list(a_counter.keys())) == list(range(25))
    assert list(a_counter.values()) == [2 for _ in range(25)]
    b_counter = Counter(key_to_elems["B"])
    assert sorted(list(b_counter.keys())) == list(range(50))
    assert list(b_counter.values()) == [1 for _ in range(50)]
    c_counter = Counter(key_to_elems["C"])
    assert sorted(list(c_counter.keys())) == list(range(6))

if __name__ == "__main__":
    test_balanced()
