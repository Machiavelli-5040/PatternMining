from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_pattern_presence_idx(
    sequence: list[int] | np.ndarray,
    pattern: list[int] | np.ndarray,
    max_gap: int,
) -> np.ndarray:
    p = len(pattern)
    sequence = np.array(sequence)
    starts = np.where(sequence == pattern[0])[0]
    presence_idx = np.ndarray((0, p), dtype=int)

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
            presence_idx = np.vstack(
                (
                    presence_idx,
                    idx_array[
                        np.array(
                            [all(np.equal(arr, pattern)) for arr in sequence[idx_array]]
                        )
                    ],
                ),
            )

    return presence_idx


def get_pattern_presence_idx_in_all_sequences(
    dataset_subfolder_mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
):
    mode_path = Path(dataset_subfolder_mode_path)
    presences = []

    for file_path in sorted(mode_path.glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = csv_df["syllable"].to_numpy()
        durations = (
            csv_df["duration"]
            if "duration" in csv_df.columns
            else np.ones_like(sequence)
        )

        presences.append(
            np.unique(get_pattern_presence_idx(sequence, pattern, max_gap))
        )

    return presences


def get_frame_coverage_from_idx_lists(
    dataset_subfolder_mode_path: str | Path,
    presence_lists=list[np.ndarray],
):
    mode_path = Path(dataset_subfolder_mode_path)
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
    dataset_subfolder_mode_path: str | Path,
    pattern: list[int] | np.ndarray,
    max_gap: int,
):
    mode_path = Path(dataset_subfolder_mode_path)
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

        presence_idx = np.unique(get_pattern_presence_idx(sequence, pattern, max_gap))
        frames_where_present += sum(durations[presence_idx])
        total_frames += sum(durations)

    return frames_where_present, total_frames


def get_most_represented_patterns(
    dataset_subfolder_mode_path: str | Path,
    pattern_file_path: str | Path,
    max_gap: int,
) -> pd.DataFrame:
    presence_frames = []
    total_frames = []
    percentage = []

    pattern_list = list(
        pd.read_csv(pattern_file_path, converters={"pattern": pd.eval})["pattern"]
    )

    for pattern in tqdm(pattern_list):
        frames, total = get_pattern_presence_in_all_sequences(
            dataset_subfolder_mode_path, pattern, max_gap
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
    coverage_idx = np.array([], dtype=int)

    for pattern in pattern_list:
        coverage_idx = np.concat(
            (
                coverage_idx,
                np.unique(get_pattern_presence_idx(sequence, pattern, max_gap)),
            )
        )

    return np.sort(np.unique(coverage_idx))


def get_sequence_coverage_data(
    dataset_subfolder_mode_path: str | Path,
    pattern_file_path: str | Path,
    max_gap: int,
):
    mode_path = Path(dataset_subfolder_mode_path)
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

        result[file_path.name] = [
            c := sum(durations[coverage_idx]),
            t := sum(durations),
            c / t,
        ]

    return result


def get_similarity_matrix_fig(
    dataset_subfolder_mode_path: str | Path,
    pattern_file_path: str | Path,
    max_gap: int,
    mode: str = "min",
):
    mode_path = Path(dataset_subfolder_mode_path)

    result_df = pd.read_csv(
        pattern_file_path, converters={"pattern": pd.eval, "sequences": pd.eval}
    )
    pattern_list = list(result_df["pattern"])
    sequence_list = list(result_df["suquences"])

    for pattern in pattern_list:
        pass
