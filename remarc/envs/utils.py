import math
import random
import numpy as np
from ..core.landscapes import Landscape


def s_solve(y):
    x = -math.log(1 / y - 1)
    return x


def generate_landscapes(
    N=5,
    sigma=0.5,
    correl=np.linspace(-1.0, 1.0, 51),
    dense=False,
    CS=False,
    num_drugs=4,
):
    A = Landscape(N, sigma, dense=dense)
    # give it two chances at this step because sometimes it doesn't converge
    try:
        Bs = A.generate_correlated_landscapes(correl)
    except Exception:
        Bs = A.generate_correlated_landscapes(correl)

    if CS:
        # this code guarantees that high-level CS will be present
        split_index = np.array_split(range(len(Bs)), num_drugs)
        keep_index = [round(np.median(i)) for i in split_index]
    else:
        keep_index = np.random.randint(0, len(Bs) - 1, size=num_drugs)

    landscapes = [Bs[i] for i in keep_index]
    drugs = [i.ls for i in landscapes]

    return landscapes, drugs


def generate_landscapes2(
    N=4, sigma=0.5, num_drugs=4, CS=False, dense=False, correl=None
):
    landscapes = []
    for i in range(num_drugs):
        landscapes.append(Landscape(N, sigma))

    drugs = [i.ls for i in landscapes]
    return landscapes, drugs


def normalize_landscapes(drugs, seascapes=False):
    if seascapes:
        for i in range(len(drugs)):
            for j in range(len(drugs[i])):
                drugs[i][j] = drugs[i][j] - np.min(drugs[i][j])
                drugs[i][j] = drugs[i][j] / np.max(drugs[i][j])
        drugs_normalized = drugs
    else:
        drugs_normalized = []
        for i in range(len(drugs)):
            drugs_i = drugs[i] - np.min(drugs[i])
            drugs_normalized.append(drugs_i / np.max(drugs_i))

    return drugs_normalized


def discretize_state(state_vector):
    S = [i for i in range(len(state_vector))]
    probs = state_vector.T.flatten()
    state = np.random.choice(S, size=1, p=probs)
    new_states = np.zeros((len(state_vector), 1))
    new_states[state] = 1
    return new_states


def fast_choice(options, probs):
    x = random.random()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]


def define_eight_state_landscapes(as_dict=False, center_at_zero=False) -> np.ndarray:
    """
    Chen et al. fitness landscapes for 8 genotypes (N=3) and 4 drugs.
    Source: Chen et al. evolutionary dynamics paper

    Parameters
    ----------
    as_dict : bool
        If True, return a dict keyed by drug name instead of a list.
    center_at_zero : bool
        If True, subtract 1.0 from all fitness values so the landscape is
        centered around 0 instead of 1.  This amplifies the signal-to-noise
        ratio for RL reward calculations while preserving the relative
        fitness differences that drive selection.
    """
    if as_dict:
        drugs = {}
        drugs["Drug_A"] = [0.993, 0.998, 1.009, 1.003, 1.007, 1.001, 0.992, 0.997]
        drugs["Drug_B"] = [0.995, 1.005, 1.002, 0.999, 1.005, 0.994, 0.999, 1.001]
        drugs["Drug_C"] = [0.997, 1.001, 0.989, 1.003, 1.003, 0.998, 1.010, 0.997]
        drugs["Drug_D"] = [1.005, 0.988, 0.999, 1.001, 0.995, 1.011, 1.000, 0.999]
    else:
        drugs = []
        drugs.append([0.993, 0.998, 1.009, 1.003, 1.007, 1.001, 0.992, 0.997])
        drugs.append([0.995, 1.005, 1.002, 0.999, 1.005, 0.994, 0.999, 1.001])
        drugs.append([0.997, 1.001, 0.989, 1.003, 1.003, 0.998, 1.010, 0.997])
        drugs.append([1.005, 0.988, 0.999, 1.001, 0.995, 1.011, 1.000, 0.999])
    result = np.array(drugs)
    if center_at_zero:
        result = result - 1.0
    return result


def define_four_state_landscapes(as_dict=False, amplification=1.0):
    """
    Four-state fitness landscapes for 4 genotypes (N=2) and 4 drugs.

    Args:
        as_dict: If True, return as dict with drug names as keys.
        amplification: Scale fitness deviations from the mean. Values > 1.0
            increase selection pressure (e.g. 10.0 turns ~1% differences
            into ~10% differences). Default 1.0 = raw values.
    """
    raw = np.array(
        [
            [0.993, 0.998, 1.009, 1.003],
            [1.005, 0.988, 0.999, 1.001],
            [0.997, 1.001, 0.989, 1.003],
            [0.995, 1.005, 1.002, 0.999],
        ]
    )

    if amplification != 1.0:
        mean = raw.mean()
        raw = mean + (raw - mean) * amplification

    if as_dict:
        drug_names = ["Drug_A", "Drug_B", "Drug_C", "Drug_D"]
        return {name: list(row) for name, row in zip(drug_names, raw)}
    return raw


def define_three_state_landscapes(as_dict=False, amplification=1.0):
    """
    Three-state fitness landscapes for 3 genotypes and 4 drugs.

    Args:
        as_dict: If True, return as dict with drug names as keys.
        amplification: Scale fitness deviations from the mean.
    """
    raw = np.array(
        [
            [0.994, 0.997, 1.009],  # Drug A
            [1.000, 0.991, 1.008],  # Drug B
            [0.999, 1.003, 0.994],  # Drug C
            [0.993, 1.005, 1.002],  # Drug D
        ]
    )

    if amplification != 1.0:
        mean = raw.mean()
        raw = mean + (raw - mean) * amplification

    if as_dict:
        drug_names = ["Drug_A", "Drug_B", "Drug_C", "Drug_D"]
        return {name: list(row) for name, row in zip(drug_names, raw)}
    return raw

def define_rmf_landscapes(N, sigma=1.0, num_drugs=4, pop_size=10000, max_Ns=0.1, seed=None, max_tries=1000):
    """
    Generates landscape of size 2^N * num_drugs.
    Returns np.array of shape (num_drugs, 2^N)
    
    """
    
    if seed is not None:
        np.random.seed(seed)
    for _ in range(max_tries):
        # Generating random landscpe
        _, drugs = generate_landscapes2(N=N, sigma=sigma, num_drugs=num_drugs)
        raw = np.array(drugs)
        # Asserting that one drug does not dominate.
        if len(set(np.argmin(raw, axis=0))) == 1:
            continue

        # Rescaling such that max distance to wt within each drug satisfies |Ns| <= max_Ns
        # gets wt genotypes
        r0 = raw[:, [0]]

        kappa = max_Ns / (pop_size * np.max(np.abs(raw-r0)))
        f = 1.0 + kappa * (raw - raw.mean())

        return f

    raise RuntimeError("no valid landscape set found")

def define_trap_landscapes(as_dict=False, amplification=1.0):
    """
    Trap fitness landscapes for 4 genotypes (N=2) and 4 drugs.
    Scaled similarly to four-state around ~1.0 with ~0.015 deviations.
    """
    raw = np.array(
        [
            [0.991, 1.005, 0.998, 1.008],  # Drug 0 (Greedy Trap)
            [1.002, 0.992, 1.005, 1.009],  # Drug 1
            [0.998, 0.995, 1.002, 1.008],  # Drug 2 (RL Setup)
            [1.005, 1.006, 0.991, 1.008],  # Drug 3 (RL Finisher)
        ]
    )

    if amplification != 1.0:
        mean = raw.mean()
        raw = mean + (raw - mean) * amplification

    if as_dict:
        drug_names = ["Drug_A", "Drug_B", "Drug_C", "Drug_D"]
        return {name: list(row) for name, row in zip(drug_names, raw)}
    return raw


def define_successful_landscapes():
    return np.array(
        [
            [1.20506869, 3.35382954, 0.32273345, 0.29391833],
            [3.99748972, 0.5007246, 1.94397629, 0.66432379],
            [2.13112861, 0.59541999, 1.32083556, 1.48314829],
            [1.0306329, 3.99639049, 0.70993479, 0.74922469],
            [2.73580711, 0.43070899, 3.9541098, 3.44483566],
            [3.00333663, 0.24205831, 2.57842129, 0.91931369],
            [1.23278399, 2.31578297, 2.80822274, 2.46058061],
            [2.80129708, 0.23133693, 0.33508322, 3.83364791],
        ]
    )
