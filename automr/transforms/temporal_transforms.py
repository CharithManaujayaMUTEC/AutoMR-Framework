# automr/transforms/temporal_transforms.py

def identity_sequence(sequence):
    return sequence


def sample_sequence(dataset, length=10):
    return dataset[:length]


def next_frame_pair(dataset, idx):
    idx = int(idx)   #  CRITICAL FIX

    if idx >= len(dataset) - 1:
        return None

    return dataset[idx], dataset[idx + 1]