from __future__ import annotations

import numpy as np

from dataset import BearingFaultDataset


def main():
    dataset = BearingFaultDataset(label_mode="binary")
    features = np.array(
        [
            [
                item.summary["BPFI_env_sum"],
                item.summary["BPFO_env_sum"],
                item.summary["BSF_env_sum"],
                item.summary["rms"],
                item.summary["kurtosis"],
                item.summary["crest"],
                item.summary["env_kurtosis"],
            ]
            for item in dataset.items
        ],
        dtype=np.float32,
    )
    labels_arr = np.array([item.label for item in dataset.items])
    healthy = features[labels_arr == 0]
    faulty = features[labels_arr == 1]

    print("Dataset size:", len(dataset))
    print("\n=== Feature Means ===")
    print("Healthy mean:", healthy.mean(axis=0))
    print("Faulty mean :", faulty.mean(axis=0))
    print("Feature vector shape:", features.shape)


if __name__ == "__main__":
    main()
