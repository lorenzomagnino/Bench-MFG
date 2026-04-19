"""Lightweight benchmark harness for core JAX MFG kernels.

Usage:
  python benchmarks/jax_core_benchmark.py --backend cpu
  python benchmarks/jax_core_benchmark.py --backend cuda --particles 128
"""

from __future__ import annotations

import argparse
import time

from envs.lasry_lions_chain.lasry_lions_chain import LasryLionsChain
from envs.lasry_lions_chain.lasry_lions_chain_jit import (
    reward_lasry_lions_chain,
    transition_lasry_lions_chain,
)
from envs.mfg_model_class_jit import (
    EnvSpec,
    Q_eval_jax,
    exploitability_batch_jax,
    get_jax_device,
    mean_field_by_transition_kernel_multi_jax,
)
import jax
import jax.numpy as jnp
import numpy as np


def _time_call(fn, *args, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        timings.append(time.perf_counter() - start)

    return min(timings), sum(timings) / len(timings)


def _build_spec() -> EnvSpec:
    n_states = 128
    n_actions = 3
    mu_0 = np.ones(n_states, dtype=np.float32) / n_states
    noise_prob = np.array([0.025, 0.95, 0.025], dtype=np.float32)
    env = LasryLionsChain(
        N_states=n_states,
        N_actions=n_actions,
        N_noises=3,
        horizon=8,
        mean_field=mu_0,
        noise_prob=noise_prob,
        crowd_penalty_coefficient=1.0,
        movement_penalty=0.1,
        center_attraction=0.5,
        gamma=0.9,
        is_noisy=True,
    )
    return EnvSpec(
        environment=env,
        transition=transition_lasry_lions_chain,
        reward=reward_lasry_lions_chain,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    device = get_jax_device(args.backend)
    spec = _build_spec()
    policy = jax.device_put(
        jnp.ones(
            (spec.environment.N_states, spec.environment.N_actions), dtype=jnp.float32
        )
        / spec.environment.N_actions,
        device,
    )
    mean_field = jax.device_put(
        jnp.asarray(spec.environment.stationary_mean_field, dtype=jnp.float32), device
    )
    policy_batch = jax.device_put(
        jnp.broadcast_to(policy, (args.particles,) + policy.shape), device
    )

    kernels = [
        (
            "mean_field_by_transition_kernel_multi_jax",
            mean_field_by_transition_kernel_multi_jax,
            (policy, spec, 20, mean_field),
        ),
        ("Q_eval_jax", Q_eval_jax, (policy, mean_field, spec)),
        (
            "exploitability_batch_jax",
            exploitability_batch_jax,
            (policy_batch, spec, mean_field, args.particles),
        ),
    ]

    print(f"device={device.platform} particles={args.particles}")
    for name, fn, fn_args in kernels:
        best, avg = _time_call(fn, *fn_args, warmup=args.warmup, repeats=args.repeats)
        print(f"{name}: best={best * 1e3:.2f}ms avg={avg * 1e3:.2f}ms")


if __name__ == "__main__":
    main()
