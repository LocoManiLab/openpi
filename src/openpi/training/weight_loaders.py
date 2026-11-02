import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """Loads weights from a checkpoint, dropping any leaf whose shape disagrees with the model.

    Use this when fine-tuning with a model config that differs from the base checkpoint in
    a few projection shapes (e.g. ``action_dim`` change). The mismatched leaves are logged
    and re-initialized from the freshly-built model; everything else loads normally.
    """

    params_path: str
    # Keys matching this regex (in flattened "/"-separated form) will *also* be re-initialized,
    # even if their shape happens to match. Useful to force a clean head when the semantics
    # changed but the dim didn't.
    force_reinit_regex: str | None = None

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        force_reinit = re.compile(self.force_reinit_regex) if self.force_reinit_regex else None

        result: dict = {}
        skipped_shape: list[str] = []
        skipped_forced: list[str] = []
        for k, v in flat_loaded.items():
            if k not in flat_ref:
                continue
            ref = flat_ref[k]
            if force_reinit is not None and force_reinit.fullmatch(k):
                skipped_forced.append(k)
                continue
            if tuple(ref.shape) != tuple(v.shape):
                skipped_shape.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(ref.shape)}")
                continue
            result[k] = v.astype(ref.dtype) if v.dtype != ref.dtype else v

        flat_loaded.clear()

        if skipped_shape:
            logger.warning(
                "PartialCheckpointWeightLoader: re-initializing %d params due to shape mismatch:\n  %s",
                len(skipped_shape),
                "\n  ".join(skipped_shape),
            )
        if skipped_forced:
            logger.warning(
                "PartialCheckpointWeightLoader: re-initializing %d params due to force_reinit_regex:\n  %s",
                len(skipped_forced),
                "\n  ".join(skipped_forced),
            )

        # Anything still missing (re-init layers + lora) keeps the freshly initialized values.
        for k in flat_ref:
            if k not in result:
                result[k] = flat_ref[k]

        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
