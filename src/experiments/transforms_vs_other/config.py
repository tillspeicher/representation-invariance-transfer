from dataclasses import dataclass
from typing import Optional

from lib_project.experiment import ExperimentHandle

from .experiment import (
    EXP_ABBREVIATION,
    ExperimentConfig,
    FineTuningFreezeConfig,
    ModelConfig,
    TrainingConfig,
    TvODataConfig,
    tvo_experiment,
)


NUM_TARGET_CLASSES = 30
IMG_SIZE = 32

N_TRAIN_EPOCHS = 50
N_FINE_TUNE_EPOCHS = {
    "linear_probe": 200,
    "full": 50,
}
CONFIG_SEED_1 = 5939
SAMPLING_SEED_1 = 29402

SEEDS = {
    0: (59832, 2032),
    1: (113, 50202),
    2: (5902, 93),
    3: (5483, 421),
    4: (78, 70887),
    5: (494, 237342),
    6: (3405, 21562),
    7: (746, 107634),
    8: (36684, 5295),
    9: (605, 34),
}

NUM_TARGET_CLASSES = {
    "t2d": 30,
    # Only use half the classes as targets for the CIFAR datasets, to simulate
    # the transfer learning setting where we train on one set of classes and
    # transfer to a different one.
    "cifar10": 5,
    "cifar100": 50,
}


def get_ft_default_sample_count(data_type: str, ft_type_id: str) -> int | float:
    if ft_type_id == "lp":
        return {
            "t2d": 50000,
            "cifar10": 1.0,
            "cifar100": 1.0,
        }[data_type]
    else:
        ft_samples = ft_type_id.split("-")[-1]
        try:
            return int(ft_samples)
        except ValueError:
            return float(ft_samples)


@dataclass
class ConfigArgs:
    data_type: str
    variation_category: str
    ft_type: str
    num_ft_samples: int | float
    learning_rate: float = 0.001
    transform_type: str = "mixed"


VARIATION_ARGS = {
    f"{data_id}_v-{variation_cat_id}_ft-{ft_type_id}": ConfigArgs(
        data_type=data_type,
        variation_category=variation_cat,
        ft_type=ft_type,
        num_ft_samples=get_ft_default_sample_count(data_type, ft_type_id),
    )
    for data_id, data_type in [
        ("t2d", "t2d"),
        ("ci10", "cifar10"),
        ("ci100", "cifar100"),
    ]
    for variation_cat_id, variation_cat in [
        ("class-rel", "class_relationship"),
        ("arch", "architecture"),
        ("pt-samples", "pretraining_samples"),
        ("ft-samples", "finetuning_samples"),
    ]
    for ft_type_id, ft_type in [
        ("lp", "linear_probe"),
        ("full-200", "full"),  # 200 samples does not fully converge
        # for disjoint transforms
        ("full-2000", "full"),  # 2000 samples converges well
        ("full-0.01", "full"),
        ("full-0.1", "full"),  # 10% of the dataset, for CIFAR
    ]
}
HYPERPARAMETER_ARGS = {
    (
        f"{data_id}_m-{model_id}_ft-{ft_type_id}"
        # # Linear probing is the base version, without full fine-tuning.
        # # We always use a learning rate of 0.001 for linear probing.
    ): ConfigArgs(
        data_type=data_type,
        variation_category="none",
        ft_type=ft_type,
        num_ft_samples=get_ft_default_sample_count(data_type, ft_type_id),
        learning_rate=0.001,
    )
    for data_id, data_type in [
        ("t2d", "t2d"),
        ("ci10", "cifar10"),
        ("ci100", "cifar100"),
    ]
    for model_id in [
        "resnet-18",
        "resnet-50",
        "densenet-121",
        "vgg-11",
        "vit",
    ]
    for ft_type_id, ft_type in [
        ("lp", "linear_probe"),
        ("full-300", "full"),  # 200 samples does not fully converge
        # for disjoint transforms
        ("full-0.1", "full"),  # 10% of the dataset, for CIFAR
    ]
}


CONFIG_ARGS = (
    {
        "test": ConfigArgs(
            data_type="t2d",
            variation_category="pretraining_samples",
            # variation_category="finetuning_samples",
            # ft_type="full",
            ft_type="linear_probe",
            num_ft_samples=get_ft_default_sample_count("t2d", "lp"),
        ),
    }
    | VARIATION_ARGS
    | HYPERPARAMETER_ARGS
)


def create_config(
    config_name: str,
    seed_id: Optional[int] = None,
) -> ExperimentConfig:
    args = CONFIG_ARGS[config_name]
    num_target_classes = NUM_TARGET_CLASSES[args.data_type]
    config = ExperimentConfig(
        name=config_name,
        seed_id=-1,
        training=TrainingConfig(
            max_epochs=N_TRAIN_EPOCHS,
            save_checkpoints=True,
        ),
        fine_tuning_training=TrainingConfig(
            max_epochs=N_FINE_TUNE_EPOCHS[args.ft_type],
            save_checkpoints=args.ft_type == "linear_probe",
            # wandb_project_name=f"rep-inv_{EXP_ABBREVIATION}",
            wandb_project_name=None,
            log_every_n_steps=1,
        ),
        fine_tuning_freeze=FineTuningFreezeConfig(
            type=args.ft_type,
            reset_head=False,
        ),
        data=get_t2d_data_config(
            data_type=args.data_type,
            transform_type=args.transform_type,
            config_seed=-1,
            sampling_seed=-1,
            num_target_classes=num_target_classes,
        ),
        num_ft_samples=args.num_ft_samples,
        model=ModelConfig(
            type="resnet-18",
            domain="cifar",
            num_classes=num_target_classes,
            init_seed=-1,
        ),
        variation_category=args.variation_category,
    )

    if config_name.startswith("test"):
        config.training.max_steps = 1
        config.fine_tuning_training.max_steps = 1
    if seed_id is not None:
        set_seeds(config, seed_id)
    return config


def get_t2d_data_config(
    data_type: str,
    transform_type: str,
    config_seed: int,
    sampling_seed: int,
    num_target_classes: int,
) -> TvODataConfig:
    return TvODataConfig(
        data_type=data_type,
        transform_type=transform_type,
        config_seed=config_seed,
        sampling_seed=sampling_seed,
        transforms_sampling_seed=None,
        num_target_classes=num_target_classes,
        img_size=IMG_SIZE,
        n_training_samples=50000,  # 1667 per class
        n_val_samples=10000,
        n_test_samples=10000,
        batch_size=512,
    )


def set_seeds(
    config: ExperimentConfig,
    seed_id: int,
) -> ExperimentConfig:
    config.seed_id = seed_id
    config_seed, sampling_seed = SEEDS[seed_id]
    config.data.config_seed = config_seed
    config.data.sampling_seed = sampling_seed
    config.data.transforms_sampling_seed = sampling_seed
    config.model.init_seed = config_seed + 583
    return config


def get_configs() -> list[ExperimentConfig]:
    configs = []
    for eval_type in CONFIG_ARGS.keys():
        config = create_config(eval_type)
        configs.append(config)
    return configs


TvOHandle = ExperimentHandle(
    id=EXP_ABBREVIATION,
    create_configs=get_configs,
    set_seed=set_seeds,
    experiment=tvo_experiment,
)
