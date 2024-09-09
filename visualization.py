from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from analysis import get_all_patterns_coverage_in_sequence
from matplotlib.colors import rgb2hex


def get_syllable_sequence_fig(
    csv_path: str | Path,
    unknown_threshold: int,
    nb_syllables: int,
    palette="gist_rainbow",
):
    """
    Returns a Plotly figure of the syllable sequence. Syllables whose corresponding
    ID is greater or equal to the `unknown_threshold` will be marked in gray.

    Works with both standard and compressed format but using the compressed one
    (with durations) is recommended (less clutter on the sequence).
    """

    csv_df = pd.read_csv(csv_path)
    syllables = csv_df["syllable"]
    durations = (
        csv_df["duration"] if "duration" in csv_df.columns else np.ones_like(syllables)
    )

    # Get colors
    cmap = plt.get_cmap(palette, unknown_threshold)
    color_seq = np.array(
        [rgb2hex(cmap(i)) for i in range(unknown_threshold)]
        + ["#bbbbbb" for _ in range(unknown_threshold, nb_syllables)]
    )

    df = pd.DataFrame({"y": np.ones_like(syllables), "duration": durations})
    fig = px.bar(
        df,
        x="duration",
        y="y",
        orientation="h",
        height=200,
        hover_data={
            "y": False,
            "pred": syllables,
        },
    )
    fig.update_traces(marker_color=color_seq[syllables])
    fig.update_xaxes(range=[0, sum(durations)])
    return fig


def get_sequence_coverage_fig(
    csv_path: str | Path,
    pattern_file_path: str | Path,
    max_gap: int,
):
    """
    Returns a Plotly figure of the syllable sequence. Syllables belonging to one
    of the patterns of the pattern_list will be highlighted in blue.

    Works with both standard and compressed format but using the compressed one
    (with durations) is recommended (less clutter on the sequence).
    """

    csv_df = pd.read_csv(csv_path)
    syllables = csv_df["syllable"].to_numpy()
    durations = (
        csv_df["duration"] if "duration" in csv_df.columns else np.ones_like(syllables)
    )
    pattern_list = list(
        pd.read_csv(pattern_file_path, converters={"pattern": pd.eval})["pattern"]
    )

    # Get pattern coverage
    pattern_coverage = get_all_patterns_coverage_in_sequence(
        syllables, pattern_list, max_gap
    )

    # Get colors
    color_seq = np.array(["#bbbbbb" for _ in range(len(syllables))])
    color_seq[pattern_coverage] = "#0000ff"

    df = pd.DataFrame({"y": np.ones_like(syllables), "duration": durations})
    fig = px.bar(
        df,
        x="duration",
        y="y",
        orientation="h",
        height=200,
        hover_data={
            "y": False,
            "pred": syllables,
        },
    )
    fig.update_traces(marker_color=color_seq[syllables])
    fig.update_xaxes(range=[0, sum(durations)])
    return fig
