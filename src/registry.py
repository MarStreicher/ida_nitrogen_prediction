from typing import Type

from models.base_model import BaseExperimentModel
from models.plsr_model import PLSRModel
from models.nn_model import SimpleNNModel

models: dict[str, Type[BaseExperimentModel]] = {
    "PLSR": PLSRModel,
    "NN": SimpleNNModel,
    # Add models
}