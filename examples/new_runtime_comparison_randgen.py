
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.agents.shepherd_eval import ShepherdMDP
from remarc.agents.tianshou_agent import get_ppo_policy, load_best_fn, train_wf_landscapes
from remarc.core.hyperparameters import Presets as P
from remarc.core.landscapes import Landscape
from remarc.envs.utils import (
    define_eight_state_landscapes,
    define_four_state_landscapes,
    define_three_state_landscapes,
)
from remarc.envs.wright_fisher_env import ThreeGenotypeEnv, WrightFisherEnv

# Direclty copied these from run plots py
LR = 1e-4
MR = 1e-5 
EPISODE_STEPS = 500
REWARD_SCALE = 100.0
AMP = 1.0
GPS = 10  
GAMMA = 0.99
ENT = 0.1
BATCH = 64
DELTA = 0.5
DELTA_HORIZON = 1
N_FRAMES = 1
STEPS_PER_EPOCH = 10000
POP_SIZE = 10000
MAX_NS = 0.1


def load_dataset(name, N=None, sigma=1.0, num_drugs=4, seed=None):
    if name == "three_state":
        data, v_N = define_three_state_landscapes(amplification=AMP), 2
    elif name == "four_state":
        data, v_N = define_four_state_landscapes(amplification=AMP), None
    elif name == "eight_state":
        data, v_N = define_eight_state_landscapes(), None
    elif name == "rmf":
        data = define_rmf_landscapes(N=N, sigma=sigma, num_drugs=num_drugs,
                                     pop_size=POP_SIZE, max_Ns=MAX_NS, seed=seed)
        v_N = None
    else:
        raise ValueError(f"Unknown dataset: {name}")

    num_drugs = data.shape[0]
    if v_N is None:
        v_N = int(np.log2(len(data[0])))
    g_min, g_max = np.min(data), np.max(data)
    reward_scale = 2.0 / (g_max - g_min) if name == "rmf" else REWARD_SCALE
    landscapes = [
        Landscape(v_N, sigma=0.0, ls=data[i], g_min=g_min, g_max=g_max)
        for i in range(num_drugs)
    ]
    return landscapes, num_drugs, v_N, g_min, g_max, reward_scale


def make_eval_env(name, landscapes, num_drugs, v_N, eval_steps, reward_scale):
    kwargs = dict(
        landscape_list=landscapes,
        num_drugs=num_drugs,
        gen_per_step=GPS,
        seq_length=v_N,
        random_start=False,
        total_generations=GPS * eval_steps + 100,
        reward_scale=REWARD_SCALE,
        stochastic=True,
        delta_multiplier=DELTA,
        mutation_rate=MR,
        n_frames=N_FRAMES,
        delta_horizon=DELTA_HORIZON,
    )
    return ThreeGenotypeEnv(**kwargs) if name == "three_state" else WrightFisherEnv(**kwargs)


def solve_mdp(env, L):
    start = time.perf_counter()
    mdp = ShepherdMDP.from_env(env, L=L, discount=GAMMA)
    mdp.solve()
    return mdp, time.perf_counter() - start


def train_rl(name, tag, env, landscapes, num_drugs, v_N, reward_scale, epochs):
    sig = f"timing_{tag}_e{epochs}"
    p = P(
        state_shape=(N_FRAMES * (2**v_N),),
        num_actions=num_drugs,
        buffer_size=20000,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=0.95,
        ent_coef=ENT,
        batch_size=BATCH,
        epochs=epochs,
        train_steps_per_epoch=STEPS_PER_EPOCH,
        reward_scale=reward_scale,
        gen_per_step=GPS,
        dataset=name,
        landscape_amplification=AMP,
        test_episodes=5,
        episode_steps=EPISODE_STEPS,
        delta_multiplier=DELTA,
        stochastic=True,
        random_start=True,
        n_frames=N_FRAMES,
        delta_horizon=DELTA_HORIZON,
    )
    object.__setattr__(p, "mutation_rate", MR)
    object.__setattr__(p, "n_train_envs", 16)
    object.__setattr__(p, "n_test_envs", 4)
    object.__setattr__(p, "hidden_sizes", [256, 256, 256])
    object.__setattr__(p, "head_sizes", [128])

    extra = {"landscapes": landscapes} if name == "rmf" else {}
    start = time.perf_counter()
    train_wf_landscapes(p, signature=sig, **extra)
    seconds = time.perf_counter() - start

    policy = get_ppo_policy(p, DummyVectorEnv([lambda: env])).eval()
    policy = load_best_fn(policy, f"best_policy_{sig}.pth")
    return policy, seconds


def effectiveness(env, choose_action, runs, steps, g_min, g_max, tail=0.2):
    finals = []
    for _ in range(runs):
        obs, _ = env.reset()
        fitness = []
        for _ in range(steps):
            fitness.append(env.get_fitness())
            obs, _, terminated, truncated, _ = env.step(choose_action(obs))
            if terminated or truncated:
                break
        k = max(1, int(len(fitness) * tail))
        finals.append(np.mean(fitness[-k:]))
    return (np.mean(finals) - g_min) / (g_max - g_min)


def make_plot(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(df))
    labels = [f"{g} genotypes" for g in df["genotypes"]]

    # Left: runtime
    ax = axes[0]
    w = 0.35
    b1 = ax.bar(x - w / 2, df["mdp_s"], w, label="MDP (SHEPHERD)")
    b2 = ax.bar(x + w / 2, df["rl_s"], w, label="RL (PPO)")
    ax.bar_label(b1, fmt="%.0f s")
    ax.bar_label(b2, fmt="%.0f s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime")
    ax.legend()

    # Right: effectiveness
    ax = axes[1]
    cols = [("fit_mdp", "MDP (SHEPHERD)"), ("fit_rl", "RL (PPO)"),
            ("fit_best_single", "Best single drug"), ("fit_random", "Random")]
    w = 0.2
    for i, (col, name) in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, df[col], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized mean fitness (lower is better)")
    ax.set_title("Effectiveness")
    ax.set_ylim(0, df[[c for c, _ in cols]].max().max() * 1.3)
    ax.legend(ncol=2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_dir / "runtime_comparison.png", dpi=200)
    print(f"Saved graph to {out_dir / 'runtime_comparison.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["three_state", "four_state", "eight_state"])
    ap.add_argument("--L", type=int, default=3, help="MDP grid points per axis (run_plots default)")
    ap.add_argument("--epochs", type=int, default=300, help="PPO epochs (run_plots uses 300)")
    ap.add_argument("--eval-runs", type=int, default=20, help="evaluation runs per policy")
    ap.add_argument("--eval-steps", type=int, default=1000, help="steps per evaluation run")
    ap.add_argument("--skip-rl", action="store_true")
    args = ap.parse_args()

    out_dir = project_root / "log" / "runtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.datasets:
        landscapes, num_drugs, v_N, g_min, g_max, reward_scale = load_dataset(name)
        env = make_eval_env(name, landscapes, num_drugs, v_N, reward_scale, args.eval_steps)
        print(f"\n=== {name}: {env.num_genotypes} genotypes, {num_drugs} drugs ===")

        print(f"Solving MDP (L={args.L})...")
        mdp, mdp_s = solve_mdp(env, args.L)
        print(f"MDP done in {mdp_s:.1f}s")

        policy, rl_s = None, float("nan")
        if not args.skip_rl:
            print(f"Training PPO ({args.epochs} epochs)...")
            policy, rl_s = train_rl(name, name, env, landscapes, num_drugs, v_N, reward_scale, args.epochs)
            print(f"PPO done in {rl_s:.1f}s")

        print("Evaluating policies...")
        n = env.num_genotypes
        ev = lambda fn: effectiveness(env, fn, args.eval_runs, args.eval_steps, g_min, g_max)

        def rl_action(obs):
            with torch.no_grad():
                return int(policy(Batch(obs=np.array([obs]), info={})).act[0])

        fit_mdp = ev(lambda obs: mdp.get_action(obs[-n:]))
        fit_rl = ev(rl_action) if policy is not None else float("nan")
        fit_random = ev(lambda obs: np.random.randint(num_drugs))
        fit_single = min(ev(lambda obs, d=d: d) for d in range(num_drugs))

        rows.append({
            "dataset": name, "genotypes": n, "drugs": num_drugs, "L": args.L,
            "epochs": args.epochs, "mdp_s": mdp_s, "rl_s": rl_s,
            "fit_mdp": fit_mdp, "fit_rl": fit_rl,
            "fit_best_single": fit_single, "fit_random": fit_random,
        })
        pd.DataFrame(rows).to_csv(out_dir / "runtime_comparison.csv", index=False)  # save as we go
        print(rows[-1])

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    make_plot(df, out_dir)
    plt.show()


if __name__ == "__main__":
    main()