from typing import Optional

from lib_project.experiment import ExperimentHandle

from .experiment import (
    EXP_ABBREVIATION,
    ExperimentConfig,
    InvarianceMeasurementConfig,
    ModelConfig,
    SeedConfig,
    STDataConfig,
    TrainingConfig,
    it_experiment,
)


NUM_CLASSES = 30

SEEDS = {
    0: SeedConfig(
        config_seed=58390,
        sampling_seed=2940,
        transforms_sampling_seeds=(593, 94224),
    ),
    1: SeedConfig(
        config_seed=9614,
        sampling_seed=2007,
        transforms_sampling_seeds=(320, 79),
    ),
    2: SeedConfig(
        config_seed=393654,
        sampling_seed=350,
        transforms_sampling_seeds=(2608, 3),
    ),
    3: SeedConfig(
        config_seed=6086,
        sampling_seed=82,
        transforms_sampling_seeds=(424997, 7955),
    ),
    4: SeedConfig(
        config_seed=79170,
        sampling_seed=83927,
        transforms_sampling_seeds=(677, 212),
    ),
    5: SeedConfig(
        config_seed=20233,
        sampling_seed=153,
        transforms_sampling_seeds=(9, 677),
    ),
    6: SeedConfig(
        config_seed=5456,
        sampling_seed=682,
        transforms_sampling_seeds=(8688, 201),
    ),
    7: SeedConfig(
        config_seed=2418,
        sampling_seed=8429,
        transforms_sampling_seeds=(7309, 66443),
    ),
    8: SeedConfig(
        config_seed=8037,
        sampling_seed=9204,
        transforms_sampling_seeds=(8933, 2735),
    ),
    9: SeedConfig(
        config_seed=8510,
        sampling_seed=178,
        transforms_sampling_seeds=(9142, 3293),
    ),
}

CONFIG_ARGS = {
    "test": "resnet-18",
    "rn18": "resnet-18",
    "dn121": "densenet-121",
    "vit": "vit",
    "vgg11": "vgg-11",
}


def create_config(
    eval_type: str,
    seed_id: Optional[int] = None,
) -> ExperimentConfig:
    model_type = CONFIG_ARGS[eval_type]
    is_test = eval_type == "test"
    config = ExperimentConfig(
        name=eval_type,
        seed_id=-1,
        seed=SeedConfig(-1, -1, (-1, -1)),
        training=TrainingConfig(
            max_epochs=50,
            save_checkpoints=True,
        ),
        fine_tuning=TrainingConfig(
            max_epochs=15,
            save_checkpoints=False,
        ),
        model=ModelConfig(
            num_classes=NUM_CLASSES,
            type=model_type,
            domain="cifar",
            init_seed=-1,
        ),
        training_data=_get_st_data_config(train=True),
        eval_data=_get_st_data_config(train=False),
        inv_measurement=InvarianceMeasurementConfig(
            metrics=["dotprod", "cos", "l2", "cka"],
            shuffle_seed=-1,
        ),
    )
    if is_test:
        config.training.max_steps = 1
        config.fine_tuning.max_steps = 1
        config.training_data.transforms = []
        config.eval_data.transforms = []
        config.eval_data.n_test_samples = config.eval_data.batch_size
    if seed_id is not None:
        set_seeds(config, seed_id)
    return config


def _get_st_data_config(
    train: bool,
) -> STDataConfig:
    if train:
        batch_size = 512
        data_types = [("op", "img"), ("rp", "rand")]
    else:
        # batch_size = 10000
        batch_size = 2048
        data_types = [
            ("op", "img"),
            ("os", "img"),
            ("op", "rand"),
            ("os", "rand"),
            ("rp", "img"),
            ("rs", "img"),
            ("rp", "rand"),
            ("rs", "rand"),
        ]
    foreground_types, background_types = zip(*data_types)
    return STDataConfig(
        n_classes=NUM_CLASSES,
        config_seed=-1,
        sampling_seed=-1,
        transforms_sampling_seed=None,
        img_size=32,
        n_training_samples=50000,  # 1667 per class
        n_val_samples=10000,
        n_test_samples=10000,
        batch_size=batch_size,
        foreground_types=list(foreground_types),
        background_types=list(background_types),
    )


def set_seeds(
    config: ExperimentConfig,
    seed_id: int,
) -> ExperimentConfig:
    seed_config = SEEDS[seed_id]
    config.seed_id = seed_id
    config.seed = seed_config
    config.eval_data.config_seed = seed_config.config_seed
    config.eval_data.sampling_seed = seed_config.sampling_seed
    config.training_data.config_seed = seed_config.config_seed
    config.training_data.sampling_seed = seed_config.sampling_seed
    config.model.init_seed = seed_config.config_seed + 394
    config.inv_measurement.shuffle_seed = seed_config.config_seed + 24
    return config


def get_configs() -> list[ExperimentConfig]:
    configs = []
    for config_name in CONFIG_ARGS.keys():
        config = create_config(config_name)
        configs.append(config)
    return configs


ITHandle = ExperimentHandle(
    id=EXP_ABBREVIATION,
    create_configs=get_configs,
    set_seed=set_seeds,
    experiment=it_experiment,
)
