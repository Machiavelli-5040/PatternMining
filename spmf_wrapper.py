from pathlib import Path

from spmf import Spmf


def VMSP(
    dataset_subfolder_mode_path: str | Path,
    min_sup: float,
    max_length: int,
    max_gap: int,
    show_sequence_ids: bool = True,
) -> Path:
    """
    Runs the VMSP algorithm from the SPMF library on the given arguments.
    Returns the path to the output file created for easier use.
    """

    mode_path = Path(dataset_subfolder_mode_path)
    output_path = (
        mode_path.parents[2].with_stem("results").joinpath(*mode_path.parts[-3:])
        / f"spmf-VMSP_minsup-{min_sup}_maxlen-{max_length}_maxgap-{max_gap}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spmf_VMSP = Spmf(
        "VMSP",
        input_filename=mode_path / "spmf_input.txt",
        output_filename=output_path,
        arguments=[min_sup / 100, max_length, max_gap, show_sequence_ids],
    )
    spmf_VMSP.run()
    return output_path
