"""Format protein files according to a jsonl list file.

Support formats:

- list (directly from list file)
- aln
- fasta
- a3m

Example Usage:

1. Transformat a3m to fasta
```
python format.py -l <list_name>.jsonl -i <input_dir> -o <output_dir> -f "a3m -> fasta"
```

2. Transformat list to fasta
```
python format.py -l <list_name>.jsonl -o <output_dir> -f "list -> fasta"
```

3. Transformat a3m to fasta without a list.
```
python format.py -i <input_dir> -o <output_dir> -f "a3m -> fasta"
```
In this situation, `format.py` will first scan the files in input directory to
build a list. Accepatable files include `.a3m`, `.fasta`, `.aln`, `.pdb`.

"""
import argparse
from pathlib import Path

from tqdm import tqdm

import lib.utils.datatool as dtool


SPLIT_SYMBOL = "->"


def get_filename(name, fmt):
    if fmt == "aln":
        suffix = ".aln"
    elif fmt == "a3m":
        suffix = ".a3m"
    elif fmt == "fasta":
        suffix = ".fasta"
    elif fmt == "jsonl":
        suffix = ".jsonl"
    else:
        raise NotImplementedError
    return f"{name}{suffix}"


def get_lines_funcs(source_format, target_format):
    if source_format == "aln":
        if target_format == "fasta":
            lines_funcs = [dtool.aln2fasta]
    elif source_format == "a3m":
        if target_format == "fasta":
            lines_funcs = [dtool.a3m2fasta]
        elif target_format == "aln":
            lines_funcs = [dtool.a3m2aln]
    elif source_format == "list":
        if target_format == "fasta":
            lines_funcs = [dtool.sample2fasta]
        elif target_format == "aln":
            lines_funcs = [dtool.sample2aln]
    else:
        raise NotImplementedError(
            f"Not supported: {source_format} {SPLIT_SYMBOL} {target_format}"
        )
    return lines_funcs


def format(
    input_dir: Path = None,
    output_dir: Path = None,
    path_list: Path = None,
    format: str = None,
):
    args = argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        list=path_list,
        format=format,
    )
    format_run(args)


def format_run(args):

    if args.list is None:
        samples = dtool.build_list_from_dir(args.input_dir)
    else:
        samples = dtool.read_jsonlines(args.list)
    source_format, target_format = args.format.split(SPLIT_SYMBOL)
    source_format, target_format = source_format.strip(), target_format.strip()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, ncols=80):
        # print(sample)
        name = sample["name"]
        """
        add segment situation
        """
        # if sample["segment"] is not None:
        #     start = sample["segment"]["start"]
        #     end = sample["segment"]["end"]
        #     name = f"{name}_s_{start}_e_{end}"
             
        output_path = Path(args.output_dir) / get_filename(name, target_format)

        if source_format == "list":
            line_func = get_lines_funcs(source_format, target_format)[0]
            lines = line_func(sample)
            dtool.write_lines(output_path, lines)
        else:
            input_path = Path(args.input_dir) / get_filename(
                name, source_format
            )
            if input_path.exists():
                dtool.process_file(
                    input_path,
                    output_path,
                    lines_funcs=get_lines_funcs(source_format, target_format),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", type=str)
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-f", "--format", required=True, type=str)

    args = parser.parse_args()
    format_run(args)
