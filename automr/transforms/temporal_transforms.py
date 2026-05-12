# automr/transforms/temporal_transforms.py

def identity_sequence(sequence):
    """
    No transformation — just pass sequence
    (used for temporal consistency testing)
    """
    return sequence


def sample_sequence(dataset, length=10):
    """
    Extract a small continuous sequence
    """
    return dataset[:length]

def next_frame_pair(dataset, idx):
    if idx >= len(dataset) - 1:
        return None
    return dataset[idx], dataset[idx + 1]