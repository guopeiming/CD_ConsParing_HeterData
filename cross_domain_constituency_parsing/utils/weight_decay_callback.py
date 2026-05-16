import numpy as np
from typing import Dict, Any, TYPE_CHECKING

from allennlp.training.callbacks.callback import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


@TrainerCallback.register("weight_decay_callback")
class WeightDecayCallback(TrainerCallback):

    def __init__(self, serialization_dir: str, gamma: int, delta: int) -> None:
        super(WeightDecayCallback, self).__init__(serialization_dir)
        self.gamma = gamma
        self.delta = delta

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary)
        epoch = 0
        # trainer.model.weight_decay(np.exp(-self.gamma*epoch/self.delta))
        trainer.model.set_grl_lambda(2/(1+np.exp(-self.gamma*epoch/self.delta))-1)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        # trainer.model.weight_decay(np.exp(-self.gamma*(epoch+1)/self.delta))
        trainer.model.set_grl_lambda(2/(1+np.exp(-self.gamma*(epoch+1)/self.delta))-1)
