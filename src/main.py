import os
import sys

import hydra


# Add the shared library to the path, else the modules in the
# library will not be able to import from each other
sys.path.insert(0, os.path.dirname(__file__) + "/lib_project")
sys.path.insert(0, os.path.dirname(__file__) + "/lib_vision")
sys.path.insert(0, os.path.dirname(__file__) + "/lib_dl_base")

from experiments.invariance_transfer.config import ITHandle
from experiments.irrelevant_feature_extraction.config import IFEHandle
from experiments.transforms_mismatch.config import TMHandle
from experiments.transforms_vs_other.config import TvOHandle
from lib_project.project import ProjectConfig, run_project, setup_project


EXPERIMENT_HANDLES = [
    TvOHandle,
    IFEHandle,
    TMHandle,
    ITHandle,
]


@hydra.main(
    version_base=None,
    config_path="../",
    config_name="project_config",
)
def main(cfg: ProjectConfig) -> None:
    run_project(
        cfg,
        EXPERIMENT_HANDLES,
    )


if __name__ == "__main__":
    setup_project(EXPERIMENT_HANDLES)
    main()
