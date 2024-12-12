from itertools import combinations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_pattern_occurences_idx_in_sequence(
    sequence: list[int] | np.ndarray,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> np.ndarray:
    """
    Returns a 2D numpy array listing all sequence indices where the pattern has been
    detected.

    Arguments:
    - `sequence: list[int] | np.ndarray` - Array or list containing the syllable
    sequence.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `np.ndarray` - Array of indices where the pattern is present in the sequence
    (indices can overlap). Shape `(N, len(pattern))` where N is the number of
    occurences of the pattern in the sequence.
    """

    p = len(pattern)
    sequence = np.array(sequence)
    starts = np.where(sequence == pattern[0])[0]
    occurences_idx = np.ndarray((0, p), dtype=int)

    for start in starts:
        idx_array = np.array(
            list(
                filter(
                    lambda seq: all(
                        ((seq[i + 1] - seq[i]) <= max_gap) for i in range(p - 1)
                    )
                    and seq[0] == start,
                    combinations(
                        range(start, min(len(sequence), start + (p - 1) * max_gap + 1)),
                        p,
                    ),
                )
            )
        )

        if idx_array.size:  # Avoid indexing error due to empty array
            occurences_idx = np.vstack(
                (
                    occurences_idx,
                    idx_array[
                        np.array(
                            [all(np.equal(arr, pattern)) for arr in sequence[idx_array]]
                        )
                    ],
                ),
            )

    return occurences_idx


def get_pattern_occurences_idx_in_all_sequences(
    mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> list[Tuple[int, np.ndarray]]:
    """
    Returns the result of get_pattern_occurences_idx_in_sequence for all sequences in a
    mode folder, grouped in a list.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `list[Tuple[int, np.ndarray]]` - List of tuples with the indices of the sequences
    and the results of get_pattern_occurences_idx_in_sequence.
    """

    mode_path = Path(mode_path)
    presences = []

    for idx, file_path in enumerate(sorted(mode_path.glob("*.csv"))):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        occurences = get_pattern_occurences_idx_in_sequence(sequence, pattern, max_gap)
        if occurences.size > 0:
            presences.append((idx, occurences))

    return presences


def get_pattern_presence_idx_in_all_sequences(
    mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> list[np.ndarray]:
    """
    Returns the indices of all frames (without duplicates) belonging to a pattern for
    all sequences in a mode folder, grouped in a list.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `list[np.ndarray]` - List of arrays of unique indices of frames belonging to the
    input pattern.
    """

    mode_path = Path(mode_path)
    presences = []

    for file_path in sorted(mode_path.glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        presences.append(
            np.unique(
                get_pattern_occurences_idx_in_sequence(sequence, pattern, max_gap)
            )
        )

    return presences


def get_frame_coverage_from_idx_lists(
    mode_path: str | Path,
    presence_lists: list[np.ndarray],
) -> Tuple[int, int]:
    """
    Returns to number of frames covered by a pattern for all sequences of a mode folder
    and their total number of frames.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `presence_lists: list[np.ndarray]` - List of arrays containing the indices of
    presence of a pattern in all sequences of the corresponding mode folder.

    Returns:
    - `Tuple[int, int]` - Total number of frames covered by a pattern and total number
    of frames of all sequences.
    """

    mode_path = Path(mode_path)
    frames_covered, total_frames = 0, 0

    for idx, file_path in enumerate(sorted(mode_path.glob("*.csv"))):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        durations = (
            csv_df["duration"]
            if "duration" in csv_df.columns
            else np.ones_like(sequence)
        )

        frames_covered += sum(durations[presence_lists[idx]])
        total_frames += sum(durations)

    return frames_covered, total_frames


def get_pattern_presence_in_all_sequences(
    mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> Tuple[int, int]:
    """
    Returns information on the number of frames covered by a pattern in the sequences of
    a mode folder.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `Tuple[int, int]` - Tuple with the total number of frames covered by `pattern` in
    all sequences of the mode folder and the total number of frames of these sequences.
    """

    mode_path = Path(mode_path)
    frames_where_present = 0
    total_frames = 0

    for file_path in sorted(mode_path.glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        durations = (
            csv_df["duration"]
            if "duration" in csv_df.columns
            else np.ones_like(sequence)
        )

        presence_idx = np.unique(
            get_pattern_occurences_idx_in_sequence(sequence, pattern, max_gap)
        )
        frames_where_present += sum(durations[presence_idx])
        total_frames += sum(durations)

    return frames_where_present, total_frames


def get_most_represented_patterns(
    mode_path: str | Path,
    pattern_file_path: str | Path,
    max_gap: int,
) -> pd.DataFrame:
    """
    Returns information on the amount of frames covered by the patterns for all
    sequences of the mode folder.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern_file_path: str | Path` - Path to the CSV file containing the patterns.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `pd.DataFrame` - Pandas DataFrame with columns `pattern` (which pattern), `frames`
    (number of frames covered), `total` (total frames of all sequences in mode folder)
    and `percentage` (percentage of frames covered by the pattern).
    """

    presence_frames = []
    total_frames = []
    percentage = []

    pattern_list = [
        arr.tolist()
        for arr in pd.read_csv(pattern_file_path, converters={"pattern": pd.eval})[
            "pattern"
        ]
    ]

    for pattern in tqdm(pattern_list):
        frames, total = get_pattern_presence_in_all_sequences(
            mode_path, pattern, max_gap
        )
        presence_frames.append(frames)
        total_frames.append(total)
        percentage.append(100 * frames / total)

    return pd.DataFrame(
        data={
            "pattern": pattern_list,
            "frames": presence_frames,
            "total": total_frames,
            "percentage": percentage,
        }
    )


def get_all_patterns_coverage_in_sequence(
    sequence: list[int] | np.ndarray,
    pattern_list: list[list[int]] | np.ndarray,
    max_gap: int,
) -> np.ndarray:
    """
    Returns information of the proportion of frames covered by at least one pattern for
    a syllable sequence.

    Arguments:
    - `sequence: list[int] | np.ndarray` - Iterable containing the sequence.
    - `pattern_list: list[list[int]] | np.ndarray` - Iterable containing the patterns.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `np.ndarray` - Array containing the indices (without duplicates) of the frames
    in `sequence` covered by at least one pattern of `pattern_list`.
    """

    coverage_idx = np.array([], dtype=int)

    for pattern in pattern_list:
        coverage_idx = np.concatenate(
            (
                coverage_idx,
                np.unique(
                    get_pattern_occurences_idx_in_sequence(sequence, pattern, max_gap)
                ),
            )
        )

    return np.unique(coverage_idx)


def get_sequence_coverage_data(
    mode_path: str | Path,
    pattern_file_path: str | Path,
    max_gap: int,
) -> dict[str, Tuple[int, int]]:
    """
    Returns information on the proportion of frames covered by at least one pattern for
    each sequence of a mode folder.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern_file_path: str | Path` - Path to the CSV file containing the patterns.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `dict[str, Tuple[int, int]]` - Dictionnary where the keys are the filenames of the
    sequences and the values are tuples with the total number of frames covered by at
    least one pattern in a sequence and the total number of frames in it.
    """

    mode_path = Path(mode_path)
    result = {}

    pattern_list = list(
        pd.read_csv(pattern_file_path, converters={"pattern": pd.eval})["pattern"]
    )

    for file_path in tqdm(sorted(mode_path.glob("*.csv"))):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        durations = (
            csv_df["duration"]
            if "duration" in csv_df.columns
            else np.ones_like(sequence)
        )
        coverage_idx = get_all_patterns_coverage_in_sequence(
            sequence, pattern_list, max_gap
        )

        result[file_path.name] = sum(durations[coverage_idx]), sum(durations)

    return result


def get_pattern_durations_in_sequence(
    sequence: list[int] | np.ndarray,
    pattern: list[int] | np.ndarray,
    durations: np.ndarray,
    max_gap: int,
) -> list[int]:
    """
    Returns a list containing the total duration (in frames) of each occurence of a
    pattern in a sequence.

    Arguments:
    - `sequence: list[int] | np.ndarray` - Iterable containing the sequence.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `durations: np.ndarray` - Array containing the durations of each syllable of the
    sequence.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `list[int]` - List of durations (in frames) of all pattern occurences in the
    sequence.
    """

    presence_array = get_pattern_occurences_idx_in_sequence(sequence, pattern, max_gap)
    duration_array = durations[presence_array].sum(axis=1)

    return duration_array.tolist()


def get_pattern_durations_in_mode(
    mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> list[int]:
    """
    Returns a list containing the total duration (in frames) of each occurence of a
    pattern in a mode folder.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `list[int]` - List of durations (in frames) of all pattern occurences in the mode
    folder.
    """

    mode_path = Path(mode_path)
    pattern_durations = []

    for file_path in sorted(mode_path.glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        durations = (
            csv_df["duration"].to_numpy()
            if "duration" in csv_df.columns
            else np.ones_like(sequence)
        )
        pattern_durations += get_pattern_durations_in_sequence(
            sequence, pattern, durations, max_gap
        )

    return pattern_durations


def plot_pattern_temporal_syllable_repartition_in_sequence(
    sequence: list[int] | np.ndarray,
    pattern: list[int] | np.ndarray,
    durations: np.ndarray,
    max_gap: int,
) -> None:
    """
    Plots a boxplot of the duration (in frames) of the syllables of a pattern across
    a sequence. One box per syllable.

    Arguments:
    - `sequence: list[int] | np.ndarray` - Iterable containing the sequence.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `durations: np.ndarray` - Array containing the durations of each syllable of the
    sequence.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.
    """

    presence_array = get_pattern_occurences_idx_in_sequence(sequence, pattern, max_gap)
    duration_array = durations[presence_array]
    n = len(pattern)
    d = len(duration_array)

    plt.figure()
    plt.boxplot(duration_array, showfliers=False)
    for i in range(n):
        plt.scatter(
            np.random.normal(i + 1, 0.04, size=d),
            duration_array[:, i],
            alpha=0.4,
        )
    plt.xticks(list(range(1, n + 1)), list(map(str, pattern)))
    plt.xlabel("Syllable")
    plt.ylabel("Duration (in frames)")
    _ = plt.show()


def plot_pattern_temporal_syllable_repartition_in_mode(
    mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> None:
    """
    Plots a boxplot of the duration (in frames) of the syllables of a pattern across
    a mode folder. One box per syllable.

    Arguments:
    - `mode_path: str | Path` - Path to the corresponding mode folder.
    - `pattern: list[int] | np.ndarray` - Array or list containing the pattern.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.
    """

    mode_path = Path(mode_path)
    n = len(pattern)
    duration_array = np.empty((0, n), dtype=int)

    for file_path in sorted(mode_path.glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        durations = (
            csv_df["duration"].to_numpy()
            if "duration" in csv_df.columns
            else np.ones_like(sequence)
        )
        presence_array = get_pattern_occurences_idx_in_sequence(
            sequence, pattern, max_gap
        )
        duration_array = np.vstack((duration_array, durations[presence_array]))

    d = len(duration_array)

    plt.figure()
    plt.boxplot(duration_array, showfliers=False)
    for i in range(n):
        plt.scatter(
            np.random.normal(i + 1, 0.04, size=d),
            duration_array[:, i],
            alpha=0.4,
        )
    plt.xticks(list(range(1, n + 1)), list(map(str, pattern)))
    plt.xlabel("Syllable")
    plt.ylabel("Duration (in frames)")
    _ = plt.show()
