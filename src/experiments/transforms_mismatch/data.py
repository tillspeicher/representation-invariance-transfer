from dataclasses import dataclass

import numpy as np

import transforms_2d as t2d


@dataclass
class TMDataConfig:
    t2d_config: t2d.Transforms2DConfig
    # Used to determine the structure of the dataset, i.e. which transforms
    # and classes to use
    config_seed: int
    n_classes: int
    n_full_transforms: int


@dataclass
class TMData:
    data: dict[str, t2d.Transforms2DData]
    transforms: list[str]


def create_tm_dataset(
    config: TMDataConfig,
) -> TMData:
    rng = np.random.default_rng(config.config_seed)

    # Subsample transforms
    n_all_transforms = config.n_full_transforms
    all_transforms = [
        str(t)
        for t in rng.choice(
            t2d.TRANSFORMS,
            n_all_transforms,
            replace=False,
        )
    ]
    print("all_transforms", all_transforms)

    # Subsample foreground objects
    n_classes = config.n_classes
    fg_indices = [
        int(c) for c in rng.choice(t2d.OBJECTS, n_classes, replace=False)
    ]
    print("fg_indices", fg_indices)

    # Create datasets with different numbers of transformations
    datasets: dict[str, t2d.Transforms2DData] = {
        f"t_{n_transforms}": t2d.create_t2d_dataset(
            config=config.t2d_config,
            transforms=all_transforms[:n_transforms],
            foregrounds=fg_indices,
        )
        for n_transforms in range(1, n_all_transforms + 1)
    }

    return TMData(
        data=datasets,
        transforms=all_transforms,
    )
