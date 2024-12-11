import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def import_results_from_kmps(results_folder_path: str | Path) -> Path:
    """
    Imports and converts Keypoint-MoSeq results to a simpler version.
    Filenames are replaced with numbers and the corresponding file order is saved in a
    JSON file called `file_name_order.json`.

    Arguments:
    - `results_folder_path: str | Path` - Path to the `results` folder of a
    Keypoint-Moseq project.

    Returns:
    - `Path` - Path to the dataset created.
    """

    # Get Keypoint-MoSeq project details
    results_path = Path(results_folder_path)
    kpms_project_name = results_path.parents[1].name
    datetime = results_path.parents[0].name

    # Create dataset folders
    dataset_path = Path.cwd() / "datasets" / f"{kpms_project_name}-{datetime}"
    full_path = dataset_path / "0" / "standard"
    full_path.mkdir(parents=True, exist_ok=True)

    file_name_order = []
    n_sequences = len(list(results_path.glob("*.csv")))
    to_pad = max(1, math.ceil(math.log10(n_sequences)))
    for idx, file_path in enumerate(sorted(results_path.glob("*.csv"))):
        file_name = file_path.name
        file_name_order.append(file_name)

        df = pd.read_csv(file_path).filter(["syllable"])
        df.to_csv(full_path / f"{str(idx).zfill(to_pad)}.csv", index=False)

    # Save dictionary
    with open(dataset_path / "file_name_order.json", "w") as file:
        json.dump(file_name_order, file, indent=4)

    # Return dataset path for other functions
    return dataset_path


def create_compressed_data(dataset_path: str | Path, force_recompute: bool = False):
    """
    Runs through all penalty subfolders of a given dataset and creates compressed
    versions for all of them, if they have not yet been created.

    Arguments:
    - `dataset_path: str | Path` - Path to the dataset folder.

    Keyword arguments:
    - `force_recompute: bool` - Optional, defaults to `False`. Whether or not to
    recompute the files that have already been computed.
    """

    data_path = Path(dataset_path)
    for subfolder in data_path.glob("*/standard/"):
        new_subfolder = subfolder.with_stem("compressed")

        if not new_subfolder.is_dir() or force_recompute:
            new_subfolder.mkdir(parents=True, exist_ok=True)

            for file_path in subfolder.glob("*.csv"):
                df = pd.read_csv(file_path)
                syllable_shifts = df["syllable"].shift() != df["syllable"]
                dur_start = np.array(df[syllable_shifts].index)
                dur_end = np.hstack((dur_start[1:], [len(df)]))
                durations = dur_end - dur_start
                df = df[syllable_shifts]
                df = df.reset_index().rename(columns={"index": "frame"})
                df.loc[:, "duration"] = durations
                df.to_csv(new_subfolder / file_path.name, index=False)
            print(
                f"Compressed folder for subfolder {subfolder.parent.name} successfully created!"
            )

        else:
            print(
                f"Compressed folder for subfolder {subfolder.parent.name} already exists, skipping."
            )


def convert_input_to_spmf(dataset_path: str | Path, force_recompute: bool = False):
    """
    Runs through all penalty subfolders of a given dataset and creates SPMF input files
    for standard and compressed modes (if they exist), if they have not yet been
    created.

    Arguments:
    - `dataset_path: str | Path` - Path to the dataset folder.

    Keyword arguments:
    - `force_recompute: bool` - Optional, defaults to `False`. Whether or not to
    recompute the SPMF input files that have already been computed.
    """

    data_path = Path(dataset_path)

    for subfolder in data_path.glob("*/"):
        for mode in ["standard", "compressed"]:
            if (subfolder / mode).is_dir():
                if (
                    not (subfolder / mode / "spmf_input.txt").is_file()
                    or force_recompute
                ):
                    with open(subfolder / mode / "spmf_input.txt", "w") as file:
                        for file_path in sorted((subfolder / mode).glob("*.csv")):
                            df = pd.read_csv(file_path)
                            syllable_list = list(df["syllable"])
                            to_write = " -1 ".join(map(str, syllable_list))
                            file.write(f"{to_write} -2\n")
                    print(
                        f"SPMF input file for subfolder {subfolder.name}/{mode} successfully created!"
                    )
                else:
                    print(
                        f"SPMF input file for subfolder {subfolder.name}/{mode} already exists, skipping."
                    )


def spmf_output_to_dataframe(spmf_output_path: str | Path):
    """
    Parses an output file of an algorithm from the SPMF library into a pandas Dataframe
    that is then saved in the same folder.

    Arguments:
    - `spmf_output_path: str | Path` - Path to the output file of the algorithm from the
    SPMF library.
    """

    spmf_path = Path(spmf_output_path)
    pattern_list, support_list, sequences_list = [], [], []
    with open(spmf_path, "r") as file:
        for line in file.read().splitlines():
            if len(line.split("#")) == 3:
                pattern, support, sequences = line.split("#")
                sequences_list.append([int(s) for s in sequences[5:].split()])
            else:
                pattern, support = line.split("#")
            pattern_list.append([int(s) for s in pattern.split(" -1 ")[:-1]])
            support_list.append(eval(support.split(":")[1]))
    df = pd.DataFrame(
        data={
            "pattern": pattern_list,
            "support": support_list,
            "sequences": sequences_list,
        }
    )
    df.to_csv(spmf_path.with_suffix(".csv"), index=False, header=True)
