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


class AttriDict(dict):
    """
    A dict which is accessible via attribute dot notation
    https://stackoverflow.com/a/41514848
    https://stackoverflow.com/a/14620633
    """

    DICT_RESERVED_KEYS = list(vars(dict).keys())

    def __init__(self, *args, **kwargs):
        """
        :param args: multiple dicts ({}, {}, ..)
        :param kwargs: arbitrary keys='value'
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        if attr not in AttriDict.DICT_RESERVED_KEYS:
            return self.get(attr)
        return getattr(self, attr)

    def __setattr__(self, key, value):
        if key == "__dict__":
            super().__setattr__(key, value)
            return
        if key in AttriDict.DICT_RESERVED_KEYS:
            raise AttributeError("You cannot set a reserved name as attribute")
        self.__setitem__(key, value)

    def __copy__(self):
        return self.__class__(self)

    def copy(self):
        return self.__copy__()


class Hiveformer(pl.LightningModule, CalvinBaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from hiveformer.models.transformer_unet import TransformerUNet

        self.model = TransformerUNet(
            hidden_size=16,
            num_layers=4,
            num_tasks=None,
            max_steps=20,
            gripper_channel=False,
            unet=False,
            use_instr_embed="all",
            instr_embed_size=512,
            num_trans_layers=1,
            nhead=8,
            txt_attn_type="self",
            num_cams=3,
            latent_im_size=(8, 8),
        )

    #! need direct dataaset, language embeddings with each sample. assert this is true
    #! translate i/o of hiveformer
    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        batch["rgb_obs"]["rgb_static"], batch["rgb_obs"]["rgb_gripper"]
        losses, logits = self.model(batch, compute_loss=True)

        return total_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            if self.state_recons:
                state_recon_loss = self.perceptual_encoder.state_reconstruction_loss()
                self.log(f"val/proprio_loss_{self.modality_scope}", state_recon_loss, sync_dist=True)
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])

            (
                sampled_plan_pp,
                action_loss_pp,
                sampled_plan_pr,
                action_loss_pr,
                kl_loss,
                mae_pp,
                mae_pr,
                gripper_sr_pp,
                gripper_sr_pr,
                seq_feat,
            ) = self.lmp_val(
                perceptual_emb, latent_goal, dataset_batch["actions"], dataset_batch["state_info"]["robot_obs"]
            )
            if "lang" in self.modality_scope:
                if self.use_bc_z_auxiliary_loss:
                    val_pred_lang_loss = self.bc_z_auxiliary_loss(
                        seq_feat, dataset_batch["lang"], dataset_batch["use_for_aux_lang_loss"]
                    )
                    self.log("val/lang_pred_loss", val_pred_lang_loss, sync_dist=True)
                if self.use_clip_auxiliary_loss:
                    val_pred_clip_loss = self.clip_auxiliary_loss(
                        seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                    )
                    self.log("val/val_pred_clip_loss", val_pred_clip_loss, sync_dist=True)
                    self.clip_groundtruth(seq_feat, dataset_batch["idx"], dataset_batch["use_for_aux_lang_loss"])
                if self.use_mia_auxiliary_loss:
                    val_pred_contrastive_loss = self.mia_auxiliary_loss(
                        seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                    )
                    self.log("val/lang_contrastive_loss", val_pred_contrastive_loss, sync_dist=True)
            val_total_act_loss_pp += action_loss_pp
            pr_mae_mean = mae_pr.mean()
            pp_mae_mean = mae_pp.mean()
            pos_mae_pp = mae_pp[..., :3].mean()
            pos_mae_pr = mae_pr[..., :3].mean()
            orn_mae_pp = mae_pp[..., 3:6].mean()
            orn_mae_pr = mae_pr[..., 3:6].mean()
            self.log(f"val_total_mae/{self.modality_scope}_total_mae_pr", pr_mae_mean, sync_dist=True)
            self.log(f"val_total_mae/{self.modality_scope}_total_mae_pp", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae_pr", pos_mae_pr, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae_pp", pos_mae_pp, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae_pr", orn_mae_pr, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae_pp", orn_mae_pp, sync_dist=True)
            self.log(f"val_kl/{self.modality_scope}_kl_loss", kl_loss, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss_pp", action_loss_pp, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss_pr", action_loss_pr, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr_pr", gripper_sr_pr, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr_pp", gripper_sr_pp, sync_dist=True)
            self.log(
                "val_act/action_loss_pp",
                val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
                sync_dist=True,
            )
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def reset(self):
        pass

    def step(self, obs, goal):
        pass

    def configure_optimizers(self):
        from hiveformer.optim.misc import build_optimizer

        cfg = self.optimizer_config

        opts = AttriDict({"optim": str(cfg.__target__).split(".")[-1], "learning_rate": cfg.lr, "betas": [0.9, 0.98]})

        optimizer = build_optimizer(self, opts)

        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")

        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
