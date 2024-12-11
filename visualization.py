from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from matplotlib.colors import rgb2hex
from pattern_analysis import get_all_patterns_coverage_in_sequence


def get_syllable_sequence_fig(
    csv_path: str | Path,
    unknown_threshold: int,
    nb_syllables: int,
    palette="gist_rainbow",
) -> plotly.graph_objs._figure.Figure:
    """
    Returns a Plotly figure of the syllable sequence. Syllables whose corresponding
    ID is greater or equal than the `unknown_threshold` will be colored in gray. Works
    with both standard and compressed formats but using the compressed one (with
    durations) is recommended (less clutter on the sequence).

    Arguments:
    - `csv_path: str | Path` - Path to the csv containing the syllable sequence to plot.
    - `unknown_threshold: int` - Threshold above which the syllables are plotted in
    gray to avoid confusion between colors.
    - `nb_syllables: int` - Maximal number of a syllable in the sequence.

    Keyword arguments:
    - `palette: str` - Optional, defaults to `"gist_rainbow"`. Matplotlib color palette
    to use for the plot.

    Returns:
    - `plotly.graph_objs._figure.Figure` - Resukting plotly figure.
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
) -> plotly.graph_objs._figure.Figure:
    """
    Returns a Plotly figure of the syllable sequence. Syllables belonging to one
    of the patterns of the pattern_list will be highlighted in blue. Works with both
    standard and compressed formats but using the compressed one (with durations) is
    recommended (less clutter on the sequence).

    Arguments:
    - `csv_path: str | Path` - Path to the csv containing the syllable sequence.
    - `pattern_file_path: str | Path` - Path to the file containing the patterns found
    for the corresponding dataset.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Returns:
    - `plotly.graph_objs._figure.Figure` - Resulting plotly figure.
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


def plot_syllable_durations_across_penalties(
    syllable: int,
    penalties: list[int],
    dataset_path: str | Path,
) -> None:
    """
    Plots the durations of a given syllable for a dataset across several penalty values.

    Arguments:
    - `syllable: int` - Syllable index to plot.
    - `penalties: list[int]` - List of penalty values to plot.
    - `dataset_path: str | Path` - Path to the corresponding dataset folder.
    """

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

    plt.figure(figsize=(10, 10))
    plt.boxplot(penalty_durations, showfliers=False)
    for i in range(n):
        plt.scatter(
            np.random.normal(i + 1, 0.04, size=len(penalty_durations[i])),
            penalty_durations[i],
            s=20,
            c=color_seq[i],
            alpha=0.5,
        )
    plt.xticks(list(range(1, n + 1)), list(map(str, penalties)))
    plt.xlabel("Penalty")
    plt.ylabel("Duration (in frames)")
    plt.title(f"Durations of syllable {syllable} across penalties")
    _ = plt.show()


def extract_gif(
    video_path: str | Path,
    output_path: str | Path,
    start_frame: int,
    duration: int,
    penalties: list[int],
    dataset_path: str | Path,
    sequence_idx: int,
    syllable_descriptions: list[str],
) -> None:
    """
        Extracts and saves a GIF file from a video. Text is added to visualize the
    syllables.

    Arguments:
    - `video_path: str | Path` - Path to the video file.
    - `output_path: str | Path` - Path to where the GIF will be saved.
    - `start_frame: int` - Starting frame of the GIF in the video.
    - `duration: int` - Number of frames to extract.
    - `penalties: list[int]` - Penalty values to write on the GIF.
    - `dataset_path: str | Path` - Path to the corresponding dataset folder.
    - `sequence_idx: int` - Sequence index of the syllable sequence corresponding to the
    video.
    - `syllable_descriptions: list[str]` - Text descriptions to add. Syllables having a
    greater index than the length of this argument will have `?` displayed instead.
    """

    dataset_path = Path(dataset_path)
    unknown_threshold = len(syllable_descriptions)
    p = len(penalties)

    sequences_list = []
    file_name = list(sorted((dataset_path / "0/compressed").glob("*.csv")))[
        sequence_idx
    ].name
    for pen in penalties:
        sequences_list.append(
            pd.read_csv(dataset_path / f"{pen}/standard/{file_name}")[
                "syllable"
            ].to_list()
        )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_list = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    ret, frame = cap.read()
    f = start_frame

    while ret:
        if f == start_frame + duration:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for i in range(p):
                syllable = sequences_list[i][f]
                # Add text for the syllables before
                frame_rgb = cv2.putText(
                    frame_rgb,
                    f"P{penalties[i]}|{syllable} ({syllable_descriptions[syllable] if syllable < unknown_threshold else "?"})",
                    org=(5, 30 * (i + 1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(204, 0, 0),
                )
                img_list.append(frame_rgb)

        ret, frame = cap.read()
        f += 1

    cap.release()
    imageio.mimsave(output_path, img_list, fps=fps, loop=0)
