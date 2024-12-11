from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from tqdm import tqdm


def get_total_syllable_frames_for_penalty(
    penalty_path: str | Path,
) -> dict[int, int]:
    """
    Returns a dictionnary giving, for each syllable, the total number of frames where it
    is present.

    Arguments:
    - `penalty_path: str | Path` - Path to the penalty folder.

    Returns:
    - `dict[int, int]` - Resulting dictionnary. Keys are the syllables, values the
    total durations in frames.
    """

    penalty_path = Path(penalty_path)
    syllable_frames = {}

    for file_path in sorted((penalty_path / "compressed").glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = list(csv_df["syllable"])
        durations = list(csv_df["duration"])

        for syllable, duration in zip(sequence, durations):
            if syllable not in syllable_frames:
                syllable_frames[syllable] = duration
            else:
                syllable_frames[syllable] += duration

    return dict(sorted(syllable_frames.items()))


def get_number_of_frame_occurences_for_penalty(
    penalty_path: str | Path,
) -> dict[int, int]:
    """
    Returns a dictionnary giving, for each syllable, its total number of occurences
    (ignoring their durations).

    Arguments:
    - `penalty_path: str | Path` - Path to the penalty folder.

    Returns:
    - `dict[int, int]` - Resulting dictionnary. Keys are the syllables, values the
     corresponding numbers of occurences.
    """

    penalty_path = Path(penalty_path)
    syllable_occurences = {}

    for file_path in sorted((penalty_path / "compressed").glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = list(csv_df["syllable"])

        for syllable in sequence:
            if syllable not in syllable_occurences:
                syllable_occurences[syllable] = 1
            else:
                syllable_occurences[syllable] += 1

    return dict(sorted(syllable_occurences.items()))
