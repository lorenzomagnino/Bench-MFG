"""Colored quick-start banners for the ``benchmfg`` CLI.

Mirrors the ``make hello``/``garnet``/``mfpso`` targets so the guides are
available after ``pip install`` too. The art matches favicon_v3.svg.
"""

from __future__ import annotations

# favicon colors + styles
DP = "\033[38;2;91;52;148m"  # dark purple  #5B3494
LP = "\033[38;2;123;82;184m"  # light purple #7B52B8
TE = "\033[38;2;26;96;128m"  # dark teal    #1A6080
GR = "\033[38;2;30;107;80m"  # dark green   #1E6B50
B = "\033[1m"  # bold
BP = "\033[1;38;2;123;82;184m"  # bold purple (titles)
BT = "\033[1;38;2;26;96;128m"  # bold teal (section headers)
R = "\033[0m"

# Ribbon fan with the "BenchMFG" wordmark to the right.
_FAN = [
    "                          " + DP + "⣀⣀⣤⣤⣶⣶⣶" + R,
    "                     " + DP + "⢀⣠⣴⣶⠿⠛⠛⠉⠉" + R,
    "                 "
    + DP
    + "⣀⣤⣴⡾⠟⠋"
    + LP
    + "⢁⣀⣤⣤⣶⣶⡾⠿⠿⠿"
    + R
    + "   "
    + BP
    + " ___              _    __  __ ___ ___ "
    + R,
    "            "
    + DP
    + "⢀⣠⣤⣶⠿⠛"
    + LP
    + "⣉⣥⣴⣶⠿⠟⠛⠉⠉"
    + R
    + "         "
    + BP
    + r"| _ ) ___ _ _  __| |_ |  \/  | __/ __|"
    + R,
    "     "
    + DP
    + "⢠⣤⣤⣴⣶⡾"
    + LP
    + "⢿⣟⣫⣭⣶⡾⠿⠛⠉"
    + TE
    + "⢁⣀⣠⣤⣤⣴⣶⣶⡾⠿⠿⠿⠿"
    + R
    + "   "
    + BP
    + r"| _ \/ -_) ' \/ _| ' \| |\/| | _| (_ |"
    + R,
    "    "
    + TE
    + "⢠⣼"
    + LP
    + "⣿⣿⣷⠿"
    + TE
    + "⣿⣟⣛⣯⣭⣤⣶⣶⠿⠿⠟⠛⠋⠉⠉⠁"
    + R
    + "          "
    + BP
    + r"|___/\___|_||_\__|_||_|_|  |_|_| \___|"
    + R,
    "     " + GR + "⢹⣿⣿⣿⣿⣛⣛⣛⣉⣉⣉⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤" + R,
    "     " + GR + "⠘⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉" + R,
]


def hello() -> None:
    """Print the BenchMFG quick-start guide."""
    print()
    print("\n".join(_FAN))
    print()
    print(f"{B}Benchmark suite for Mean Field Game algorithms.{R}")
    print()
    print(f"{BT}Initialize the repository:{R}")
    print("  uv sync --extra dev")
    print()
    print(f"{BT}List what is registered:{R}")
    print("  benchmfg env list")
    print("  benchmfg algo list")
    print()
    print(f"{BT}Environments:{R}")
    print("  contraction_game, four_rooms_obstacles, kinetic_congestion")
    print("  lasry_lions_chain, MF_Garnet, multiple_equilibria")
    print("  no_interaction_game, potential_game2d, rock_paper_scissors, sis_epidemic")
    print()
    print(f"{BT}Algorithms:{R}")
    print(
        "  damped_fixed_point (pure fixed point, damped fixed point, fictitious play, "
        "), omd, pi (policy iteration, smooth_policy_iteration, "
        "boltzmann_policy_itration), MFPso"
    )
    print()
    print(f"{BP}First example:{R}")
    print("  benchmfg train algorithm=pso environment=kinetic_congestion device=cpu")
    print()
    print(f"{BP}First sweep:{R}")
    print("  benchmfg sweep algorithm=omd environment=lasry_lions_chain \\")
    print("    experiment.name=omd_sweep experiment.random_seed=42,10 \\")
    print("    algorithm.omd.learning_rate=0.5,0.05")
    print()
    print(f"{BP}After runs:{R}")
    print("  benchmfg plot single-run <run_dir>")
    print("  benchmfg plot sweep <environment> <algorithm>")
    print("  benchmfg plot compare <environment>")


_GARNET_BANNER = [
    r" __  __ _____      ____    _    ____  _   _ _____ _____ ",
    r"|  \/  |  ___|    / ___|  / \  |  _ \| \ | | ____|_   _|",
    r"| |\/| | |_ _____| |  _  / _ \ | |_) |  \| |  _|   | |  ",
    r"| |  | |  _|_____| |_| |/ ___ \|  _ <| |\  | |___  | |  ",
    r"|_|  |_|_|        \____/_/   \_\_| \_\_| \_|_____| |_|  ",
]


def garnet() -> None:
    """Print MF-Garnet usage notes."""
    print()
    print(BP + "\n".join(_GARNET_BANNER) + R)
    print()
    print("MF-Garnet generates controlled random MFG instances for benchmarking.")
    print("Fix environment.reward.mfgarnet.seed to reproduce one game instance.")
    print()
    print(f"{BT}What varies:{R}")
    print("  - num_states, num_actions, branching_factor")
    print("  - dynamics_structure: additive | multiplicative")
    print("  - reward_structure: additive | multiplicative")
    print("  - game_type: potential | cyclic")
    print()
    print(f"{BP}Small run:{R}")
    print("  benchmfg train environment=mf_garnet algorithm=omd device=cpu \\")
    print("    environment.num_states=5 environment.num_actions=5 \\")
    print("    environment.reward.mfgarnet.seed=0 \\")
    print("    environment.reward.mfgarnet.branching_factor=5 \\")
    print("    environment.reward.mfgarnet.dynamics_structure=additive \\")
    print("    environment.reward.mfgarnet.reward_structure=multiplicative \\")
    print("    algorithm.omd.num_iterations=20")
    print()
    print(f"{BT}Benchmark protocol:{R}")
    print("  - vary environment.reward.mfgarnet.seed for game instances")
    print("  - vary experiment.random_seed for algorithm initialization")
    print("  - keep size and hyperparameters fixed inside a comparison")
    print()
    print(f"{BP}Batch helpers:{R}")
    print("  benchmfg garnet scaling --states 20 80 130 400 --no-plots")
    print("  benchmfg garnet aggregate outputs")
    print("  benchmfg garnet plot-scaling outputs")
    print("  ./scripts/garnet/run_garnet_omd.sh")
    print("  ./scripts/garnet/run_garnet_pso.sh")
    print()
    print(f"{BT}Full notes:{R} docs/MFG_GARNET.md")


_MFPSO_BANNER = [
    r" __  __ _____ ____  ____   ___  ",
    r"|  \/  |  ___|  _ \/ ___| / _ \ ",
    r"| |\/| | |_  | |_) \___ \| | | |",
    r"| |  | |  _| |  __/ ___) | |_| |",
    r"|_|  |_|_|   |_|   |____/ \___/ ",
]


def mfpso() -> None:
    """Print Mean Field PSO usage notes."""
    print()
    print(BP + "\n".join(_MFPSO_BANNER) + R)
    print()
    print("Registered as algorithm=pso.")
    print()
    print(f"{BT}Main idea:{R}")
    print("  Particle Swarm Optimization searches directly over policy logits.")
    print("  Each particle is converted to a policy, evaluated by exploitability,")
    print("  then moved by inertia, cognitive best, and swarm best terms.")
    print()
    print(f"{BT}Main characteristics:{R}")
    print("  - derivative-free search over policies")
    print("  - vectorized JAX exploitability evaluation")
    print("  - CUDA path is selected when the configured JAX device is a GPU")
    print("  - key knobs: num_particles, num_iterations, temperature, w, c1, c2")
    print()
    print(f"{BP}Small example:{R}")
    print("  benchmfg train algorithm=pso environment=kinetic_congestion device=cpu \\")
    print("    algorithm.pso.num_particles=40 \\")
    print("    algorithm.pso.num_iterations=30 \\")
    print("    algorithm.pso.temperature=0.2 \\")
    print("    algorithm.pso.w=0.4 algorithm.pso.c1=0.5 algorithm.pso.c2=1.5")
    print()
    print(f"{BP}Sweep example:{R}")
    print("  benchmfg sweep algorithm=pso environment=kinetic_congestion \\")
    print("    experiment.name=pso_sweep experiment.random_seed=42,10 \\")
    print("    algorithm.pso.w=0.3,0.7 algorithm.pso.c1=0.3,0.7 \\")
    print("    algorithm.pso.c2=0.6,1.2 algorithm.pso.temperature=0.2,0.7")
    print()
    print(f"{BP}Scripted sweep:{R}")
    print("  ./scripts/run_pso.sh")


if __name__ == "__main__":
    hello()
