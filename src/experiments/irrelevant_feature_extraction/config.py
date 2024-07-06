from hydra.core.config_store import ConfigStore

from lib_project.experiment import ExperimentHandle

from .experiment import (
    EXP_ABBREVIATION,
    ExperimentConfig,
    IFEDataConfig,
    ModelConfig,
    TrainingConfig,
    ife_experiment,
)


N_CLASSES = 10
N_TRAIN_EPOCHS = 200
N_FINE_TUNE_EPOCHS = 100

SEEDS = {
    0: (111, 694),
    1: (222, 8320),
    2: (9600, 2817),
    3: (7894, 303),
    4: (7754, 5486),
    5: (3058, 9525),
    6: (5351, 1807),
    7: (5159, 6003),
    8: (3877, 7693),
    9: (7334, 2649),
}

CONFIG_VALUES = {
    "test": ("resnet-18", ["none"]),
    "rn18_fixed": ("resnet-18", ["none"]),
    "rn18_translate": ("resnet-18", ["translate"]),
    "vgg11_fixed": ("vgg-11", ["none"]),
    "dense121_fixed": ("densenet-121", ["none"]),
}


def register_configs() -> None:
    cs = ConfigStore.instance()

    for config_name in CONFIG_VALUES.keys():
        cs.store(
            name=config_name,
            group=EXP_ABBREVIATION,
            node=create_config(config_name),
        )


def create_config(
    config_name: str,
) -> ExperimentConfig:
    model_type, transforms = CONFIG_VALUES[config_name]
    config = ExperimentConfig(
        name=config_name,
        seed_id=-1,
        training=TrainingConfig(
            max_epochs=N_TRAIN_EPOCHS,
            save_checkpoints=True,
        ),
        fine_tuning=TrainingConfig(
            max_epochs=N_FINE_TUNE_EPOCHS,
            save_checkpoints=True,
        ),
        data=_get_obj2d_data_config(),
        model=ModelConfig(
            type=model_type,
            domain="cifar",
            num_classes=N_CLASSES,
            init_seed=-1,
        ),
        transforms=transforms,
        all_models=False,
    )
    if config_name == "test":
        config.training.max_steps = 1
        config.fine_tuning.max_steps = 1
    return config


def set_seeds(
    config: ExperimentConfig,
    seed_id: int,
) -> ExperimentConfig:
    config_seed, sampling_seed = SEEDS[seed_id]
    config.seed_id = seed_id
    config.data.config_seed = config_seed
    config.data.sampling_seed = sampling_seed
    config.model.init_seed = config_seed + 493
    return config


def _get_obj2d_data_config() -> IFEDataConfig:
    return IFEDataConfig(
        config_seed=-1,
        sampling_seed=-1,
        transforms_sampling_seed=None,
        n_classes=N_CLASSES,
        img_size=32,
        n_training_samples=50000,  # 1667 per class
        n_val_samples=10000,
        n_test_samples=10000,
        batch_size=512,
    )


def get_configs() -> list[ExperimentConfig]:
    configs = []
    for eval_type in CONFIG_VALUES.keys():
        config = create_config(eval_type)
        configs.append(config)
    return configs


IFEHandle = ExperimentHandle(
    id=EXP_ABBREVIATION,
    create_configs=get_configs,
    set_seed=set_seeds,
    experiment=ife_experiment,
)
