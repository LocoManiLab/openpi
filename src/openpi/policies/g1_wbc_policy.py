"""Input/Output transforms for the G1 heft and amo whole-body controllers.

Sibling of ``g1_sonic_policy``. lab collects with three controllers on the same robot,
and each records a different action space, so each needs its own pack/unpack:

  sonic  state 43  action 78 = motion_token(64) + hands(7+7)      -> g1_sonic_policy
  heft   state 29  action 50 = ref_qpos(36) + hands(14)           -> HEFT here
  amo    state 32  action 36 = one flat teleop command vector     -> AMO here

Both are the same three steps as the sonic module -- concatenate the named action streams
on the way in, slice them back out on the way out -- so rather than copy that file twice
this one is driven by an ``ActionLayout``. sonic keeps its own module: it is a proven path
with trained checkpoints pointing at it, and there is nothing to gain by churning it.

Only the ego camera is real; pi0/pi05 expect three image slots, so the two wrist slots are
zero-filled and masked off, exactly as the sonic module does.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class ActionLayout:
    """Named action streams, in the order they concatenate into the action vector.

    ``streams`` is (name, dim); ``name`` is the key the repack transform delivers under
    ``data["actions"]`` and the key inference returns. The sum must equal the controller's
    action dim, which is what the host's adapter expects to receive.
    """

    streams: tuple[tuple[str, int], ...]

    @property
    def dim(self) -> int:
        return sum(d for _, d in self.streams)


# heft: the tracking policy's reference frame + Dex3 hand targets. Kept as ref_qpos(36)
# rather than pre-split into root_pos/root_quat/joint_pos because that is how the dataset
# stores it; the deploy server is what splits it into the three keys the sim's
# RefQposAdapter requires.
HEFT_LAYOUT = ActionLayout(streams=(("ref_qpos", 36), ("hands", 14)))

# amo: arms 14 + rpy 3 + height 1 + torso_vx/vy/vyaw + target_yaw, plus the hands in
# their own stream. Two streams, matching the dataset: as of schema v5 amo's hands live
# in `action.hands` like every other backend's, so its `action` column is 22 rather than
# the 36 it was when the hands were its first 14 entries. The split into per-group keys
# happens at deploy.
AMO_LAYOUT = ActionLayout(streams=(("action", 22), ("hands", 14)))


@dataclasses.dataclass(frozen=True)
class DeployContract:
    """What a policy server must READ from and WRITE to the host, per controller.

    Training only needs the ActionLayout; deploy needs two more things, and getting either
    wrong fails in a way the policy cannot report:

    ``state_groups``  which ``state.*`` entries to concatenate, IN DATASET ORDER, to
        rebuild the proprio vector the model was trained on. The host sends a dict, so a
        wrong order is not a crash -- it is a silently scrambled input.
    ``action_keys``   (sim key, source stream, slice) -- how to cut the policy's output
        into the keys the host's adapter requires (g1_backend/sim2real/vla/adapters.py).
        A missing key IS caught, by ``check_action_keys`` on the first frame.
    """

    state_groups: tuple[str, ...]
    action_keys: tuple[tuple[str, str, slice], ...]

    @property
    def state_dim_hint(self) -> str:
        return "+".join(self.state_groups)


# sonic keeps its own module/server path; listed here so the three contracts can be read
# side by side and so a server can dispatch over one table.
SONIC_DEPLOY = DeployContract(
    state_groups=("left_leg", "right_leg", "waist", "left_arm",
                  "left_hand", "right_arm", "right_hand"),          # 43
    action_keys=(
        ("action.motion_token", "motion_token", slice(None)),
        ("action.left_hand_joints", "left_hand_joints", slice(None)),
        ("action.right_hand_joints", "right_hand_joints", slice(None)),
    ),
)

# heft: the sim's RefQposAdapter requires the reference frame as THREE separate keys, not
# the combined 36-D ref_qpos the dataset stores, plus 7+7 hand joint targets.
HEFT_DEPLOY = DeployContract(
    state_groups=("left_leg", "right_leg", "waist", "left_arm", "right_arm"),   # 29
    action_keys=(
        ("action.ref_root_pos", "ref_qpos", slice(0, 3)),
        ("action.ref_root_quat", "ref_qpos", slice(3, 7)),
        ("action.ref_joint_pos", "ref_qpos", slice(7, 36)),
        ("action.left_hand_joints", "hands", slice(0, 7)),
        ("action.right_hand_joints", "hands", slice(7, 14)),
    ),
)

# amo: VelocityCmdAdapter requires torso_vx/vy/vyaw; the rest are optional but the
# controller tracks them, so emit everything the flat command vector carries.
# rpy/height are NOT joints -- the host measures them (waist joints / pelvis z) and sends
# them as state.rpy / state.height.
AMO_DEPLOY = DeployContract(
    state_groups=("left_hand", "right_hand", "left_arm", "right_arm", "rpy", "height"),  # 32
    action_keys=(
        ("action.left_arm", "action", slice(0, 7)),
        ("action.right_arm", "action", slice(7, 14)),
        ("action.rpy", "action", slice(14, 17)),
        ("action.height", "action", slice(17, 18)),
        ("action.torso_vx", "action", slice(18, 19)),
        ("action.torso_vy", "action", slice(19, 20)),
        ("action.torso_vyaw", "action", slice(20, 21)),
        ("action.target_yaw", "action", slice(21, 22)),
        # Out of the flat vector and into their own stream in schema v5, so
        # these offsets are now the same 0:7 / 7:14 every other backend uses.
        # The emitted key is `*_hand_joints` because that is what the host's
        # Sim2RealBackend.vla_hand_targets looks for -- `action.left_hand` was
        # accepted too, but only as a compatibility spelling.
        ("action.left_hand_joints", "hands", slice(0, 7)),
        ("action.right_hand_joints", "hands", slice(7, 14)),
    ),
)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_g1_wbc_example(state_dim: int) -> dict:
    """Random example matching the post-repack input format (testing / inference dummy)."""
    return {
        "state": np.random.rand(state_dim).astype(np.float32),
        "image": {"ego_view": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8)},
        "prompt": "open the fridge door in front of you",
    }


@dataclasses.dataclass(frozen=True)
class G1WbcInputs(transforms.DataTransformFn):
    """Pack repacked LeRobot fields into the standard pi0/pi05 observation dict."""

    model_type: _model.ModelType
    layout: ActionLayout

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

        actions = data.get("actions")
        if actions is not None:
            # A single-stream layout arrives as the bare array, not a dict of one.
            if len(self.layout.streams) == 1 and not isinstance(actions, dict):
                inputs["actions"] = np.asarray(actions, dtype=np.float32)
            else:
                inputs["actions"] = np.concatenate(
                    [np.asarray(actions[name], dtype=np.float32)
                     for name, _ in self.layout.streams],
                    axis=-1,
                )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1WbcOutputs(transforms.DataTransformFn):
    """Split the model output back into the named action streams (inference only)."""

    layout: ActionLayout

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])  # (action_horizon, model_action_dim)
        actions = actions[..., : self.layout.dim]
        out: dict = {}
        offset = 0
        for name, dim in self.layout.streams:
            out[name] = actions[..., offset : offset + dim]
            offset += dim
        return out
