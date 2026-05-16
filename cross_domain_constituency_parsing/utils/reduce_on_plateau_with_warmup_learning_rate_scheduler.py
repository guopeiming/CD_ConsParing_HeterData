from allennlp.training.learning_rate_schedulers import LearningRateScheduler, ReduceOnPlateauLearningRateScheduler
from allennlp.training.optimizers import Optimizer
from typing import Union, List
import logging


logger = logging.getLogger(__name__)


@LearningRateScheduler.register("reduce_on_plateau_with_warmup")
class ReduceOnPlateauWithWarmupLearningRateScheduler(ReduceOnPlateauLearningRateScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold_mode: str = "rel",
        threshold: float = 0.0001,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8
    ) -> None:
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]
            self.lr = param_group["lr"]

        super(ReduceOnPlateauWithWarmupLearningRateScheduler, self).__init__(
            optimizer, mode, factor, patience, verbose, threshold_mode, threshold, cooldown, min_lr, eps)

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total <= self.warmup_steps:
            factor = batch_num_total / self.warmup_steps

            for param_group in self.lr_scheduler.optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * factor

    def step(self, metric: float = None) -> None:
        if self.lr_scheduler.last_epoch > 0:
            lr = self.lr_scheduler._last_lr[0]
        else:
            lr = self.lr

        super().step(metric)

        new_lr = self.lr_scheduler._last_lr[0]
        if lr != new_lr and self.verbose:
            msg = "Epoch %.5d: reducing learning rate from %.4e to %.4e." % (self.lr_scheduler.last_epoch, lr, new_lr)
            logger.info(msg)
