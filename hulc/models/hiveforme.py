import logging
import math
from typing import Any, Dict
import sys

sys.path.append("/home/ubuntu/hulc-1")

from hiveformer.models.transformer_unet import TransformerUNet
from calvin_agent.models.calvin_base_model import CalvinBaseModel
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torchvision
import pytorch3d.transforms as t
from einops import asnumpy, rearrange as rea


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

def chw(t):
    return rea(t, 'h w c -> c h w')

def distance_map_to_point_cloud(distances, fov, width, height):
    """Converts from a depth map to a point cloud.
    Args:
        distances: An numpy array which has the shape of (height, width) that
        denotes a distance map. The unit is meter.
        fov: The field of view of the camera in the vertical direction. The unit
        is degrees.
        width: The width of the image resolution of the camera.
        height: The height of the image resolution of the camera.
    Returns:
        point_cloud: The converted point cloud from the distance map. It is a numpy
        array of shape (height, width, 3).
    """
    fov = math.radians(fov)
    f = height / (2 * math.tan(fov / 2.0))
    px = torch.tile(torch.arange(width), [height, 1]).to(distances)
    x = (2 * (px + 0.5) - width) / f * distances / 2    
    py = torch.tile(torch.arange(height), [width, 1]).T.to(distances)
    y = (2 * (py + 0.5) - height) / f * distances / 2    
    point_cloud = torch.stack((x, y, distances), dim=-1)
    return point_cloud

def plot_3d(t):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from einops import asnumpy
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y, z = asnumpy(t)
    ax.scatter3D(x, y, z, c=z, cmap='viridis')
    plt.savefig('test.png')

def invert_extrinsic_matrix(matrix):
    # returns NOT HOMOGENOUS COORDINATES. returns 3x4
    # Assuming matrix is a 4x4 tensor
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_transpose = R.t()
    t_new = -torch.mm(R_transpose, t.unsqueeze(1)).squeeze()
    # inv_matrix = torch.eye(4).to(matrix)
    return torch.concat((R_transpose, t_new[:, None]), dim=-1) # 3 x 4



def point_cloud(distances, fov, width, height, inv_extrinsic):
    bsz = distances.shape[0]
    pc = distance_map_to_point_cloud(distances, fov, width, height)  # (B, L, H, W, XYZ)    
    # add homo
    rea_pc = rea(pc, "B L H W XYZ -> (B L) XYZ (H W)")
    L, _, HW = rea_pc.shape
    ones = torch.ones((L, 1, HW)).to(rea_pc)
    rea_pc_homo = torch.concat((rea_pc, ones), dim=-2)
    # to world coords
    rea_inv_extrinsic = rea(inv_extrinsic, "B L T F -> (B L) T F")

    rea_ret = torch.bmm(rea_inv_extrinsic, rea_pc_homo)
    return rea(rea_ret, "(B L) XYZ (H W) -> B L XYZ H W", B=bsz, H=height)

def compute_view_matrix(look_from, look_at, up_vector):
    import torch.nn.functional as F
    # 1. Compute the forward, right, and up vectors for the camera.
    z_axis = F.normalize((look_from - look_at), dim=0)   # Forward vector
    x_axis = F.normalize(torch.cross(up_vector, z_axis), dim=0)  # Right vector
    y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)     # True up vector

    # 2. Construct the rotation matrix using these vectors.
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)

    # 3. Construct the translation matrix using the look_from vector.
    translation = -torch.matmul(rotation.transpose(0, 1), look_from.unsqueeze(1))

    # 4. Combine the rotation and translation matrices to get the view matrix.
    view_matrix = torch.eye(4)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = translation.squeeze()

    return view_matrix

class Resize(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.resizer = torchvision.transforms.Resize(200)

    def forward(self, inp):
        bsz = inp.shape[0]
        resized = self.resizer(rea(inp, "B L C H W -> (B L) C H W"))
        return rea(resized, "(B L) C H W -> B L C H W", B=bsz)


class Hiveformer(pl.LightningModule, CalvinBaseModel):
    def __init__(self, lr_scheduler, optimizer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = TransformerUNet(
            hidden_size=16,
            num_layers=4,
            num_tasks=None,
            max_steps=20,
            gripper_channel=False,
            unet=False,
            use_instr_embed="all",
            instr_embed_size=384,
            num_trans_layers=1,
            nhead=8,
            txt_attn_type="self",
            num_cams=3,
            latent_im_size=(8, 8),
        )
        self.resize200 = Resize(200)
        self.lr_scheduler = lr_scheduler
        self.optimizer_config = optimizer
        self.lang_embeddings = None
        self.register_buffer(
            "static_extrinsic",
            invert_extrinsic_matrix(
                torch.tensor(
                    [
                        [0.72324455, -0.03026059, 0.68992877, 0.21526623],
                        [0.51412338, 0.69061059, -0.50865924, -0.26317155],
                        [-0.46107975, 0.72259355, 0.51503795, -4.39916897],
                        [0., 0., 0., 1.0],
                    ]
                )
            ),
        )
        #! need direct dataaset, language embeddings with each sample. assert this is true

    def preprocess(self, batch):
        batch = batch["lang"]
        pos_orn = batch["state_info"]["robot_obs"][:6].float()
        camera_pos, camera_orn = pos_orn[..., :3], pos_orn[..., 3:6]  # (B, L, 3)
        cam_rot = t.euler_angles_to_matrix(camera_orn, "XYZ")  # (B, L, 3, 3)
        gripper_extrinsic = torch.concat((cam_rot, camera_pos[..., None]), dim=-1)
        homo = torch.zeros_like(gripper_extrinsic)[:, :, :1]
        homo[:, :, :, -1] = 1
        
        gripper_extrinsic = torch.concat((gripper_extrinsic, homo), dim=-2)

        from functorch import vmap
        rea_gripper_extrinsic = vmap(invert_extrinsic_matrix)(rea(gripper_extrinsic, "B L H W -> (B L) H W"))

        bsz, length, *_ = batch["depth_obs"]["depth_static"].shape
        gripper_pc = point_cloud(batch["depth_obs"]["depth_gripper"], 
                                 75, 84, 84, rea(rea_gripper_extrinsic, "(B L) H W -> B L H W", B=bsz))
        static_pc = point_cloud(
            batch["depth_obs"]["depth_static"], 10, 200, 200, self.static_extrinsic.expand(bsz, length, -1, -1)
        )


        torch.save(asnumpy(static_pc[0, 0][:, ::5, ::5]), 'static_pc.pt')
        torch.save(asnumpy(gripper_pc[0, 0][:, ::2, ::2]), 'gripper_pc.pt')
        breakpoint()
        plot_3d(static_pc[0, 0])
        depth_static = np.uint8((asnumpy(batch["depth_obs"]["depth_static"][0, 0]) + 1) / 2 * 255)
        # rgb_static = np.uint8(rea((asnumpy(batch["rgb_obs"]["rgb_static"][0, 0]) + 1) / 2 * 255, 'c h w -> h w c' ))
        # from PIL import Image
        # Image.fromarray(rgb_static).save('testing.png')
        breakpoint()
        model_input = AttriDict(
            rgbs=torch.stack((batch["rgb_obs"]["rgb_static"], self.resize200(batch["rgb_obs"]["rgb_gripper"])), dim=2),
            pcds=torch.stack((static_pc, self.resize200(gripper_pc)), dim=2),
            step_masks=torch.ones(
                1, batch["rgb_obs"]["rgb_static"].size(1)
            ).long(),  #! check. just masking out the null steps
            instr_embeds=batch["lang"],
            txt_masks=torch.ones(1, 1),
            taskvar_ids=None,
            step_ids=None,
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
            self.log(f"train/{k}", losses[k])

        return losses["total"]

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
            self.log(f"val/{k}", losses[k])
        return losses

    def reset(self):
        pass

    def step(self, obs, goal):
        if isinstance(goal, str):
            embedded_lang = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()

        obs["language"] = embedded_lang
        return self.model(self.preprocess(obs))  #! check out

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

        opts = AttriDict(
            {
                "optim": str(cfg._target_).split(".")[-1].lower(),
                "learning_rate": cfg.lr,
                "betas": [0.9, 0.98],
                "weight_decay": cfg.weight_decay,
            }
        )  #! rm betas in future

        optimizer = build_optimizer(self, opts)
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}


if __name__ == "__main__":
    s = Hiveformer(None, None)
    s.to("cuda")
    batch = {"lang": torch.load("/home/ubuntu/hulc-1/test_batch.pt")}
    s.training_step(batch, 0)
