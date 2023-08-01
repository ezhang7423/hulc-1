import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

from calvin_agent.models.calvin_base_model import CalvinBaseModel
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from hulc.models.decoders.action_decoder import ActionDecoder
from hulc.utils.distributions import State

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class Hiveformer(pl.LightningModule, CalvinBaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)