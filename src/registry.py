from typing import Type

from models.base_model import BaseExperimentModel
from models.plsr_model import PLSRModel

models: dict[str, Type[BaseExperimentModel]] = {
    "PLSR": PLSRModel,
    # Add models
}