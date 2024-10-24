from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from tqdm import tqdm


def get_total_syllable_frames_for_penalty(
    mode_path: str | Path,  # compressed
):
    mode_path = Path(mode_path)
    syllable_frames = {}

    for file_path in sorted(mode_path.glob("*.csv")):
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
    mode_path: str | Path,  # compressed
):
    mode_path = Path(mode_path)
    syllable_occurences = {}

    for file_path in sorted(mode_path.glob("*.csv")):
        csv_df = pd.read_csv(file_path)
        sequence = list(csv_df["syllable"])

        for syllable in sequence:
            if syllable not in syllable_occurences:
                syllable_occurences[syllable] = 1
            else:
                syllable_occurences[syllable] += 1

    return dict(sorted(syllable_occurences.items()))
