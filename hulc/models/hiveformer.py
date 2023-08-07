import logging
from typing import Any, Dict

from calvin_agent.models.calvin_base_model import CalvinBaseModel
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import pybullet as p
import torchvision


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

# from hiveformer/preprocess/generate_dataset_keysteps.py:35, rlbench/utils.py:257, PyRep/pyrep/objects/vision_sensor.py

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

def pointcloud_from_depth_and_camera_params(
            depth: np.ndarray, extrinsics: np.ndarray,
            intrinsics: np.ndarray) -> np.ndarray:
        """Converts depth (in meters) to point cloud in word frame.
        :return: A numpy array of size (width, height, 3)
        """
        upc = _create_uniform_pixel_coords_image(depth.shape)
        pc = upc * np.expand_dims(depth, -1)
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(intrinsics, extrinsics)
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = np.expand_dims(_pixel_to_world_coords(
            pc, cam_proj_mat_inv), 0)
        world_coords = world_coords_homo[..., :-1][0]
        return world_coords
    
    
class Hiveformer(pl.LightningModule, CalvinBaseModel):
    def __init__(self, lr_scheduler, *args: Any, **kwargs: Any) -> None:
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
        self.resize200 = torchvision.transforms.Resize(200)
        self.lr_scheduler = lr_scheduler
        #! need direct dataaset, language embeddings with each sample. assert this is true

    def preprocess(self, batch): 
        breakpoint()
        pos_orn = batch['state_info']['robot_obs'][:6]
        camera_pos, camera_orn = pos_orn[:3], pos_orn[3:]
        # from calvin_env_repo/calvin_env/calvin_env/camera/gripper_camera.py        
        cam_rot = p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]

        view_matrix = p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y, -cam_rot_z)
        projection_matrix = p.computeProjpectionMatrixFOV(
            fov=self.fov, aspect=self.aspect, nearVal=self.nearval, farVal=self.farval
        )        
        static_pc = pointcloud_from_depth_and_camera_params(batch["depth_obs"]['depth_static'], [], [])
        gripper_pc = pointcloud_from_depth_and_camera_params(batch["depth_obs"]['depth_gripper'], view_matrix, projection_matrix)
        
        
        model_input = AttriDict(
            rgbs=torch.stack((batch["rgb_obs"]["rgb_static"], self.resize200(batch["rgb_obs"]["rgb_gripper"])), dim=2),
            pc_obs=torch.stack((static_pc, gripper_pc), dim=2),
            step_masks=torch.ones(1, batch["rgb_obs"]["rgb_static"].size(1)).long(),  #! check. just masking out the null steps
            instr_embeds=batch['language'],
            txt_masks=torch.ones(1, batch['language'].size(1)).long(), #! check. just masking out the null lang
        )
        return model_input
    
    
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

        The robot proprioceptive information, which also includes joint positions can be accessed with:

        ['rel_actions']
        (dtype=np.float32, shape=(7,))
        tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
        tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
        gripper_action (1): binary (close = -1, open = 1)

        ['robot_obs']
        (dtype=np.float32, shape=(15,))
        tcp position (3): x,y,z in world coordinates
        tcp orientation (3): euler angles x,y,z in world coordinates
        gripper opening width (1): in meter
        arm_joint_states (7): in rad
        gripper_action (1): binary (close = -1, open = 1)
        
        Returns:
            loss tensor
        """

        losses, logits = self.model(self.preprocess(batch), compute_loss=True)
        for k in losses:
            self.log(f'train/{k}', losses[k])

        return losses['total']

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
        losses, logits = self.model(self.preprocess(batch), compute_loss=True)
        for k in losses:
            self.log(f'val/{k}', losses[k])
        return losses


    def reset(self):
        pass

    def step(self, obs, goal):
        obs['language'] = goal        
        return self.model(self.preprocess(obs)) #! check out

    def configure_optimizers(self):
        """
        optim: 'adamw'
        learning_rate: 5e-4
        lr_sched: 'linear' # inverse_sqrt, linear
        betas: [0.9, 0.98]
        weight_decay: 0.001
        grad_norm: 5
        dropout: 0.1
        train_batch_size: 16
        gradient_accumulation_steps: 1
        num_epochs: null
        num_train_steps: 300000
        warmup_steps: 2000
        log_steps: 1000
        save_steps: 5000        
        """
             
        from hiveformer.optim.misc import build_optimizer

        cfg = self.optimizer_config

        opts = AttriDict({"optim": str(cfg.__target__).split(".")[-1], "learning_rate": cfg.lr, "betas": [0.9, 0.98]}) #! rm betas in future

        optimizer = build_optimizer(self, opts)
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
h