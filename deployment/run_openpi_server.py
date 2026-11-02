#!/usr/bin/env python3
"""pi0.5 (openpi) policy server speaking Isaac-GR00T's ZMQ protocol (:5550 / get_action).

openpi ships only its own websocket server (`scripts/serve_policy.py`), but LocoManiLab's real
robot loop and the tele_humanoid_benchmark sim client speak the gr00t REQ/REP format.
This adapter wraps `openpi.policies.policy.Policy` so a pi0.5 checkpoint is evaluated by
exactly the same client as gr00t / fastwam / starVLA / PRTS -- which is what makes the
success rates comparable.

Mirrors starVLA's starvla_gr00t_server.py and LocoManiLab's prts_gr00t_server.py; keep all three
in sync if the wire format changes.

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REQ/REP
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config

Action layout (78-D, identical to the other G1 SONIC models):
    action.motion_token(64) + action.left_hand_joints(7) + action.right_hand_joints(7)
G1SonicOutputs already splits the model output into exactly these three streams.

Usage (from the openpi checkout, under `uv run`):
    uv run python run_openpi_server.py \
        --config-name pi05_g1_sonic_toilet401 \
        --ckpt-path checkpoints/pi05_g1_sonic_toilet401/<exp>/20000 \
        --port 5550 --initial-prompt "walk over to the toilet and stand next to it"
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import numpy as np

MOTION_TOKEN_DIM = 64
HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM      # 78


class OpenPiPolicy:
    """Adapts openpi's Policy.infer() to the gr00t get_action() contract."""

    def __init__(self, policy, initial_prompt: Optional[str] = None,
                 upsample_factor: int = 1, contract=None):
        self.policy = policy
        self._current_prompt = initial_prompt
        self.upsample_factor = max(1, int(upsample_factor))
        # Which controller this checkpoint drives: decides both the proprio order we
        # rebuild and the action keys we emit. Defaults to sonic so an older invocation
        # behaves exactly as before.
        from openpi.policies import g1_wbc_policy as _wbc
        self.contract = contract or _wbc.SONIC_DEPLOY

    # ---- gr00t endpoints ---------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": "openpi (pi0.5) gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(self, observation: dict[str, Any],
                   options: dict[str, Any] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
        t0 = time.monotonic()

        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt, call reset(prompt=...), or include "
                "obs['language']['annotation.human.task_description'][0][0].")
        self._current_prompt = prompt

        frame = self._extract_image(observation)          # [H,W,3] uint8
        state = self._extract_proprio(observation)        # [43] float32
        if state is None:
            raise RuntimeError("pi0.5 needs proprio state (43-D); none found in observation.")

        # Shape expected by G1SonicInputs (it does the resize/normalisation itself).
        obs = {"image": {"ego_view": frame}, "state": state, "prompt": prompt}

        t_inf = time.monotonic()
        out = self.policy.infer(obs)
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        # The Outputs transform already split the model output into named streams; the
        # contract says how those map onto the keys this controller's adapter requires.
        streams = {name: self._upsample(np.asarray(arr, dtype=np.float32))
                   for name, arr in out.items()
                   if isinstance(arr, np.ndarray) or np.ndim(arr) > 0}
        action_dict = {}
        for sim_key, stream, sl in self.contract.action_keys:
            if stream not in streams:
                raise KeyError(
                    f"policy output has no stream {stream!r} (has {sorted(streams)}) — "
                    f"checkpoint and --config-name disagree about the action space?")
            action_dict[sim_key] = np.ascontiguousarray(
                streams[stream][:, sl], dtype=np.float32)

        horizon = int(next(iter(action_dict.values())).shape[0])
        print(f"[pi05] get_action: H_out={horizon} (x{self.upsample_factor}); "
              f"infer={infer_ms:.0f} ms; prompt={prompt!r}", flush=True)
        return action_dict, {"inference_time_s": time.monotonic() - t0,
                             "chunk_horizon": horizon}

    # ---- internals ---------------------------------------------------------
    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        v = lang.get("annotation.human.task_description")
        while isinstance(v, (list, tuple, np.ndarray)) and len(v):
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        return v if isinstance(v, str) and v else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        """Newest ego-view frame as HWC uint8.

        The tele sim client sends `img[np.newaxis, np.newaxis]`, i.e. (1, 1, H, W, 3) --
        FIVE dims, under the key "ego_view" (not "video.ego_view"). Real-robot callers may
        send (T, H, W, 3) or bare (H, W, 3). Peel every leading singleton/time axis instead
        of assuming a fixed rank: getting this wrong surfaces far downstream as
        `permute(sparse_coo): input.dim() = 5 is not equal to len(dims) = 3`.
        """
        video = obs.get("video") or {}
        for key in ("video.ego_view", "ego_view", "video.webcam", "webcam"):
            arr = video.get(key) if key in video else obs.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            while arr.ndim > 3:          # drop batch/time axes, keeping the newest frame
                arr = arr[-1]
            if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
                raise RuntimeError(f"ego-view image has unexpected shape {np.shape(arr)}; "
                                   f"expected (..., H, W, 3)")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        raise RuntimeError(f"No ego-view image in observation; keys={list(obs)} "
                           f"video={list(video)}")

    # The 43-D G1 state is these joint groups in THIS order (deploy_living_room_vla.py
    # _STATE_SLICES). Do not sort the dict: alphabetical order scrambles the vector, and
    # "projected_gravity" is an EXTRA field that must not be concatenated into the 43.
    # Kept for reference; the live order comes from self.contract.state_groups.
    STATE_ORDER = ("left_leg", "right_leg", "waist", "left_arm",
                   "left_hand", "right_arm", "right_hand")

    def _extract_proprio(self, obs: dict) -> Optional[np.ndarray]:
        state = obs.get("state") or {}
        if not state:
            return None
        order = self.contract.state_groups
        missing = [k for k in order if k not in state]
        if missing:
            raise KeyError(
                f"observation is missing state.{{{','.join(missing)}}} which this "
                f"checkpoint's proprio vector needs ({self.contract.state_dim_hint}). "
                f"Present: {sorted(state)}")
        keys = list(order)
        parts = []
        for k in keys:
            v = np.asarray(state[k], dtype=np.float32)
            while v.ndim > 1:          # client sends (1, 1, D); keep the newest row
                v = v[-1]
            parts.append(v.ravel())
        return np.concatenate(parts).astype(np.float32) if parts else None

    def _upsample(self, chunk: np.ndarray) -> np.ndarray:
        if self.upsample_factor == 1 or chunk.shape[0] < 2:
            return chunk
        t_in = np.arange(chunk.shape[0], dtype=np.float32)
        t_out = np.linspace(0, chunk.shape[0] - 1,
                            chunk.shape[0] * self.upsample_factor, dtype=np.float32)
        return np.stack([np.interp(t_out, t_in, chunk[:, d])
                         for d in range(chunk.shape[1])], axis=1).astype(np.float32)


def serve(policy: OpenPiPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[pi05] PolicyServer ready on tcp://{host}:{port}", flush=True)

    while True:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                sock.send(mnp.packb({"status": "ok"}))
                break
            elif endpoint == "reset":
                resp = policy.reset(data)
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                action, info = policy.get_action(data.get("observation", data),
                                                 data.get("options"))
                # The sim client does `action, _info = resp[0], resp[1]` -- it expects a
                # LIST, not a dict. Returning the action dict directly makes the client die
                # with a misleading `KeyError: 0` while the server logs look perfectly fine.
                resp = [action, info]
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
            sock.send(mnp.packb(resp))
        except zmq.error.ContextTerminated:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                sock.send(mnp.packb({"error": str(e)}))
            except Exception:
                pass
    print("[pi05] server stopped", flush=True)


def load_policy(args) -> OpenPiPolicy:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config, args.ckpt_path, default_prompt=args.initial_prompt)
    contract = deploy_contract_for(train_config)
    print(f"[pi05] loaded {args.config_name} from {args.ckpt_path} "
          f"(state {contract.state_dim_hint}; emits {len(contract.action_keys)} action keys)",
          flush=True)
    return OpenPiPolicy(policy, args.initial_prompt, args.upsample_factor, contract)


def deploy_contract_for(train_config):
    """Which controller a TrainConfig drives, from its data factory.

    Derived rather than passed as a flag: the config already fully determines the action
    space, and a flag that can disagree with the checkpoint is a way to fail silently.
    """
    from openpi.policies import g1_wbc_policy as _wbc
    from openpi.training import config as _config

    factory = type(train_config.data).__name__
    table = {
        "LeRobotG1SonicDataConfig": _wbc.SONIC_DEPLOY,
        "LeRobotG1HeftDataConfig": _wbc.HEFT_DEPLOY,
        "LeRobotG1AmoDataConfig": _wbc.AMO_DEPLOY,
    }
    if factory not in table:
        raise SystemExit(
            f"{train_config.name}: data factory {factory} has no deploy contract — add one "
            f"to g1_wbc_policy (known: {', '.join(table)}).")
    return table[factory]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="pi0.5 (openpi) policy server (gr00t-compatible, :5550/get_action).")
    p.add_argument("--config-name", required=True,
                   help="openpi TrainConfig name, e.g. pi05_g1_sonic_toilet401")
    p.add_argument("--ckpt-path", required=True,
                   help="checkpoint dir, e.g. checkpoints/<config>/<exp>/20000")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--upsample-factor", type=int, default=1)
    args = p.parse_args(argv)

    serve(load_policy(args), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
