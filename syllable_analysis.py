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


def plot_syllable_durations_across_penalties(
    syllable: int,
    penalties: list[int],
    dataset_path: str | Path,
):
    penalty_durations = []
    n = len(penalties)

    for pen in penalties:
        pen_durations = []
        mode_path = Path(dataset_path) / f"{pen}" / "compressed"

        for file_path in sorted(mode_path.glob("*.csv")):
            csv_df = pd.read_csv(file_path)
            durations = csv_df[csv_df["syllable"] == syllable]["duration"].to_list()
            pen_durations += durations

        penalty_durations.append(pen_durations)

    cmap = plt.get_cmap("gist_rainbow", n)
    color_seq = np.array([rgb2hex(cmap(i)) for i in range(n)])

    plt.figure()
    plt.boxplot(penalty_durations, showfliers=False)
    for i in range(n):
        plt.scatter(
            np.random.normal(i + 1, 0.04, size=len(penalty_durations[i])),
            penalty_durations[i],
            c=color_seq[i],
            alpha=0.4,
        )
    plt.xticks(list(range(1, n + 1)), list(map(str, penalties)))
    plt.xlabel("Penalty")
    plt.ylabel("Duration (in frames)")
    plt.title(f"Durations of syllable {syllable} across penalties")
    _ = plt.show()
