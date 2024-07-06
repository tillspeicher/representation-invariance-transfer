from dataclasses import dataclass

import transforms_2d as t2d
from lib_project.experiment import ExperimentHandle

from .experiment import (
    EXP_ABBREVIATION,
    ExperimentConfig,
    FineTuningFreezeConfig,
    ModelConfig,
    TMDataConfig,
    TrainingConfig,
    tm_experiment,
)


N_CLASSES = 30
N_FULL_TRANSFORMS = 8
IMG_SIZE = 32

N_TRAIN_EPOCHS = 100
N_FINE_TUNE_EPOCHS = 200

SEEDS = {
    0: (9825, 507031),
    1: (174, 348779),
    2: (148, 7367),
    3: (1219, 901),
    4: (696, 3787),
    5: (673, 6150),
    6: (469, 4611),
    7: (6995, 1756),
    8: (4360, 264),
    9: (1750, 4345),
}


@dataclass
class ConfigsArgs:
    model_type: str


CONFIG_ARGS = {
    "test": ConfigsArgs("resnet-18"),
    "rn18": ConfigsArgs("resnet-18"),
    "vgg11": ConfigsArgs("vgg-11"),
    "dn121": ConfigsArgs("densenet-121"),
    "vit": ConfigsArgs("vit"),
}


def create_config(
    config_name: str,
    seed_id: int | None = None,
) -> ExperimentConfig:
    print("config_name", config_name)
    args = CONFIG_ARGS[config_name]
    config = ExperimentConfig(
        name=config_name,
        seed_id=-1,
        training=TrainingConfig(
            max_epochs=N_TRAIN_EPOCHS,
            save_checkpoints=True,
        ),
        fine_tuning_training=TrainingConfig(
            max_epochs=N_FINE_TUNE_EPOCHS,
            save_checkpoints=False,
        ),
        fine_tuning_freeze=FineTuningFreezeConfig(type="linear_probe"),
        data=TMDataConfig(
            t2d_config=t2d.Transforms2DConfig(
                sampling_seed=-1,
                img_size=IMG_SIZE,
                n_training_samples=50000,  # 1667 per class
                n_val_samples=10000,
                n_test_samples=10000,
                batch_size=256,
            ),
            config_seed=-1,
            n_classes=N_CLASSES,
            n_full_transforms=N_FULL_TRANSFORMS,
        ),
        model=ModelConfig(
            type=args.model_type,
            domain="cifar",
            num_classes=N_CLASSES,
            init_seed=-1,
        ),
    )
    if config_name.startswith("test"):
        config.data.n_full_transforms = 2
        config.training.max_steps = 1
        config.fine_tuning_training.max_steps = 1
    if seed_id is not None:
        config = set_seeds(config, seed_id)

    return config


def set_seeds(
    config: ExperimentConfig,
    seed_id: int,
) -> ExperimentConfig:
    config_seed, sampling_seed = SEEDS[seed_id]
    config.seed_id = seed_id
    config.data.config_seed = config_seed
    config.data.t2d_config.sampling_seed = sampling_seed
    config.model.init_seed = config_seed + 53
    return config


def get_configs() -> list[ExperimentConfig]:
    configs = []
    for eval_type in CONFIG_ARGS.keys():
        config = create_config(eval_type)
        configs.append(config)
    return configs


TMHandle = ExperimentHandle(
    id=EXP_ABBREVIATION,
    create_configs=get_configs,
    set_seed=set_seeds,
    experiment=tm_experiment,
)
