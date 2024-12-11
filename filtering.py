from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit
from sklearn.preprocessing import OneHotEncoder


@njit
def get_best_previous_state_and_soc(
    soc_vec: np.ndarray, end_state: int, penalty: float
) -> Tuple[int, float]:
    """
    Updates the sum of costs (soc) and the best previous state given the end state and a penalty.

    Arguments:
    - `soc_vec: np.ndarray` - Vector of sums of costs (soc), shape `(n_states,)`.
    - `end_state: int` - The end state for which the update is performed.
    - `penalty: float` - Penalty value for the update.

    Returns:
    - `Tuple[int, float]` - Tuple containing the best previous state and the updated sum of costs.
    """

    n_states = soc_vec.shape[0]
    best_previous_state = end_state
    best_soc = soc_vec[best_previous_state]
    for k_state in range(n_states):
        if k_state != end_state:
            soc = soc_vec[k_state]
            if soc + penalty < best_soc:
                best_previous_state = k_state
                best_soc = soc + penalty
    return best_previous_state, best_soc


@njit
def get_state_sequence(
    sorted_syllables: np.ndarray,
    costs: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """
    Returns the optimal state sequence for a given cost array and penalty.

    Arguments:
    - `sorted_syllables: np.ndarray` - Array of syllables present in the symbolic sequence, sorted (output of `np.unique` on the symbolic sequence).
    - `costs: np.ndarray` - Array of cost values, shape `(n_samples, n_states)`.
    - `penalty: float` - Penalty value.

    Returns:
    - `np.ndarray` - Optimal state sequence, shape `(n_samples,)`.
    """

    n_samples, n_states = costs.shape
    soc_array = np.empty((n_samples + 1, n_states), dtype=np.float64)
    state_array = np.empty((n_samples + 1, n_states), dtype=np.int32)
    soc_array[0] = 0
    state_array[0] = -1

    # Forward loop
    for end in range(1, n_samples + 1):
        for k_state in range(n_states):
            best_state, best_soc = get_best_previous_state_and_soc(
                soc_vec=soc_array[end - 1], end_state=k_state, penalty=penalty
            )
            soc_array[end, k_state] = best_soc + costs[end - 1, k_state]
            state_array[end, k_state] = best_state

    # Backtracking
    end = n_samples
    state = np.argmin(soc_array[end])
    states = np.empty(n_samples, dtype=np.int32)
    while (state > -1) and (end > 0):
        states[end - 1] = sorted_syllables[state]
        state = state_array[end, state]
        end -= 1
    return states


def get_filtered_signal(
    dataset_path: str | Path,
    penalty: int,
    distance_matrix: np.ndarray | None = None,
) -> None:
    """
    Creates and saves the optimal state sequence from a dataset. If no distance matrix
    is provided, all distances between symbols are set to 1.

    Arguments:
    - `dataset_path: str | Path` - Path to the dataset.
    - `penalty: int` - Penalty value.

    Keyword arguments:
    - `distance_matrix: np.ndarray | None` - Optional, defaults to `None`. Distance
    matrix to use for the filtering.
    """

    data_path = Path(dataset_path)
    filtered_path = data_path / f"{penalty}" / "standard"
    filtered_path.mkdir(parents=True, exist_ok=True)
    for file_path in (data_path / "0" / "standard").glob("*.csv"):
        df = pd.read_csv(file_path)
        symbols = df["syllable"].to_numpy()
        sorted_present_symbols = np.unique(symbols)

        if distance_matrix is None:
            ohe = OneHotEncoder(dtype=np.float32).fit(symbols.reshape(-1, 1))
            signal = np.asarray(ohe.transform(symbols.reshape(-1, 1)).todense())
            costs = 1 - signal
        else:
            dist_mat_present = distance_matrix[:, sorted_present_symbols]
            costs = dist_mat_present[symbols]

        filtered_signal = get_state_sequence(
            sorted_syllables=sorted_present_symbols, costs=costs, penalty=penalty
        )
        pd.DataFrame(filtered_signal, columns=["syllable"]).to_csv(
            filtered_path / file_path.name, index=False
        )
