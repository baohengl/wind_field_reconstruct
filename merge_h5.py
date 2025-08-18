import argparse
import glob
import os


def merge_h5_files(input_dir: str, output_file: str, pattern: str = "windfield_all_cases*.h5") -> None:
    """Merge multiple HDF5 files produced by data_collection.py into one file.

    Parameters
    ----------
    input_dir: str
        Directory containing ``windfield_all_cases*.h5`` files.
    output_file: str
        Path to the merged HDF5 output file.
    pattern: str, optional
        Glob pattern to match input HDF5 files inside ``input_dir``.
    """

    import h5py

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in {input_dir}"
        )

    print("Merging files:")
    for f in files:
        print("  -", f)

    with h5py.File(output_file, "w") as h5_out:
        for fpath in files:
            with h5py.File(fpath, "r") as h5_in:
                for case in h5_in.keys():
                    if case in h5_out:
                        raise ValueError(
                            f"Duplicate case '{case}' found in {fpath}"
                        )
                    h5_in.copy(h5_in[case], h5_out, name=case)
    print(f"Merged file written to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple wind field HDF5 files")
    parser.add_argument("input_dir", help="Directory containing windfield_all_cases*.h5 files")
    parser.add_argument("output_file", help="Output path for merged HDF5 file")
    parser.add_argument(
        "--pattern",
        default="windfield_all_cases*.h5",
        help="Glob pattern for input files (default: windfield_all_cases*.h5)",
    )
    args = parser.parse_args()

    merge_h5_files(args.input_dir, args.output_file, args.pattern)


if __name__ == "__main__":
    main()
