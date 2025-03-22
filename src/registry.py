from typing import Type

from models.base_model import BaseExperimentModel
from models.plsr_model import PLSRModel
from models.nn_model import SimpleNNModel
from models.bottleneck_model import BottleneckNNModel
from models.cnn_model import CNNModel
from models.autoencoder import AutoencoderModel

models: dict[str, Type[BaseExperimentModel]] = {
    "PLSR": PLSRModel,
    "NN": SimpleNNModel,
    "BN": BottleneckNNModel,
    "CNN": CNNModel,
    "AEC": AutoencoderModel
    # Add models
}