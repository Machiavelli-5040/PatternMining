from pathlib import Path

from spmf import Spmf


def VMSP(
    mode_path: str | Path,
    min_sup: float,
    max_length: int,
    max_gap: int,
    show_sequence_ids: bool = True,
) -> Path:
    """
    Runs the VMSP algorithm from the SPMF library on the given arguments.

    Arguments:
    - `mode_path: str | Path` - Path to the mode folder.
    - `min_sup: float` - Minimum support value to be used by VMSP.
    - `max_length: int` - Maximum pattern length value to be used by VMSP.
    - `max_gap: int` - Maximum distance (in frames) between two consecutive items of a
    pattern.

    Keyword arguments:
    - `show_sequence_ids: bool` - Optional, defaults to `True`. Whether or not to show
    to sequence IDs where the patterns are present.

    Returns:
    - `Path` - Path to the output file of the algorithm.
    """

    mode_path = Path(mode_path)
    output_path = (
        mode_path.parents[2].with_stem("results").joinpath(*mode_path.parts[-3:])
        / f"spmf-VMSP_minsup-{min_sup}_maxlen-{max_length}_maxgap-{max_gap}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spmf_VMSP = Spmf(
        "VMSP",
        input_filename=str(mode_path / "spmf_input.txt"),
        output_filename=str(output_path),
        arguments=[min_sup / 100, max_length, max_gap, show_sequence_ids],
    )
    spmf_VMSP.run()
    return output_path
