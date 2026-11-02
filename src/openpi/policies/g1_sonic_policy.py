"""Input/Output transforms for the G1 Sonic VLA dataset.

The dataset is a humanoid (Unitree G1) trajectory in LeRobot v2.0 format with:
  - one egocentric RGB camera   : observation.images.ego_view (480, 640, 3)
  - 43-dim joint state          : observation.state
  - 64-dim Sonic motion token   : action.motion_token
  - 7-dim Dex3 left hand joints : teleop.left_hand_joints
  - 7-dim Dex3 right hand joints: teleop.right_hand_joints

Action = concat(motion_token, left_hand, right_hand) => 78 dim per timestep.
The repack transform in `LeRobotG1SonicDataConfig` packs the three action streams
into a nested dict under the ``actions`` key; this module concatenates them.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Action layout produced by `G1SonicInputs` (used by `G1SonicOutputs` for slicing).
MOTION_TOKEN_DIM = 64
LEFT_HAND_DIM = 7
RIGHT_HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM  # 78


def make_g1_sonic_example() -> dict:
    """Random example matching the post-repack input format. Used for testing/inference dummy."""
    return {
        "state": np.random.rand(43).astype(np.float32),
        "image": {
            "ego_view": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "prompt": "take the cup and walk forward and put it in basket",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1SonicInputs(transforms.DataTransformFn):
    """Pack repacked LeRobot fields into the standard pi0/pi05 observation dict."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        ego = _parse_image(data["image"]["ego_view"])
        zeros = np.zeros_like(ego)

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": ego,
                "left_wrist_0_rgb": zeros,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        actions_dict = data.get("actions")
        if actions_dict is not None:
            inputs["actions"] = np.concatenate(
                [
                    np.asarray(actions_dict["motion_token"], dtype=np.float32),
                    np.asarray(actions_dict["left_hand"], dtype=np.float32),
                    np.asarray(actions_dict["right_hand"], dtype=np.float32),
                ],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1SonicOutputs(transforms.DataTransformFn):
    """Split the model output back into the three action streams (inference only)."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])  # (action_horizon, model_action_dim)
        actions = actions[..., :ACTION_DIM]
        end_motion = MOTION_TOKEN_DIM
        end_left = end_motion + LEFT_HAND_DIM
        end_right = end_left + RIGHT_HAND_DIM
        return {
            "motion_token": actions[..., :end_motion],
            "left_hand_joints": actions[..., end_motion:end_left],
            "right_hand_joints": actions[..., end_left:end_right],
        }
