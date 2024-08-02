"""Input and output files."""

import concurrent.futures
from pathlib import Path

import polars as pl


def collapse_strands(bed):
    """Collapse strands."""
    pos = bed.filter(pl.col("strand") == "+")
    neg = bed.filter(pl.col("strand") == "-").with_columns(
        (pl.col("start")).alias("cStart")
    )  # add column for start site on complementary strand

    joint = pos.join(neg, left_on=["chr", "end"], right_on=["chr", "cStart"], how="full", coalesce=True).with_columns(
        pl.concat_str([pl.col("strand"), pl.col("strand_right")], separator="/", ignore_nulls=True)
    )

    merged = (
        joint.with_columns(
            pl.min_horizontal("start", "start_right").alias("start"),
            pl.max_horizontal("end", "end_right").alias("end"),
            pl.col(["percent_methylated_right", "coverage_right", "percent_methylated", "coverage"])
            .fill_null(0)
            .cast(pl.UInt64),
        )
        .with_columns((pl.col("coverage") + pl.col("coverage_right")).alias("total_coverage"))
        .filter(pl.col("total_coverage") > 0)
        .with_columns(
            (
                (
                    pl.col("coverage") * pl.col("percent_methylated")
                    + pl.col("coverage_right") * pl.col("percent_methylated_right")
                )
                / pl.col("total_coverage")
            ).alias("avg")
        )  # calculated weighted average
    )

    return merged


def read_files(file, mincov, collapse):
    """Scan files."""
    name = Path(file).stem
    print(f"Scanning {name}")
    bed = (
        pl.scan_csv(
            file,
            separator="\t",
            skip_rows=1,
            has_header=False,
            dtypes={
                "column_1": pl.Utf8,
                "column_2": pl.UInt64,
                "column_3": pl.UInt64,
                "column_6": pl.Utf8,
                "column_10": pl.UInt64,
                "column_11": pl.UInt64,
            },
        )  # cannot use scan() on zipped file
        .select(
            ["column_1", "column_2", "column_3", "column_6", "column_10", "column_11"]
        )  # only select relevant columns
        .rename(
            {
                "column_1": "chr",
                "column_2": "start",
                "column_3": "end",
                "column_6": "strand",
                "column_10": "coverage",
                "column_11": "percent_methylated",
            }
        )  # rename to something that makes more sense
        .with_columns(
            pl.col("chr").str.replace_all(r"(?i)Chr", "")  # remove "chr" from Chr column to match reference
        )
    )

    if collapse:
        data = collapse_strands(bed)
        data = data.with_columns(
            maxQuant=pl.col("total_coverage").quantile(0.999, "nearest")
        )  # calculate max coverage quantile
    else:
        data = bed.with_columns(
            pl.col("percent_methylated").alias("avg"),
            pl.col("coverage").alias("total_coverage"),
            maxQuant=pl.col("total_coverage").quantile(0.999, "nearest"),  # calculate max coverage quantile
        )

    quants = data.with_columns(
        over=pl.col("total_coverage") - pl.col("maxQuant")
    )  # identify rows that go over 99 quantile

    data_cov_filt = (
        quants.filter((pl.col("total_coverage") >= mincov) & (pl.col("over") < 0))  # filter by coverage
        .with_columns(pl.lit(name).alias("sample"))
        .select(["chr", "start", "end", "strand", "avg", "sample"])
        .cast({"chr": pl.Utf8, "start": pl.UInt64, "avg": pl.Float64, "sample": pl.Utf8})
    )

    return data_cov_filt


def save_files(df, outpath):
    """Save files w/o streaming."""
    filename = (
        df.unique(subset="sample", keep="any")
        .select(pl.col("sample").filter(pl.col("sample") != "imputed").first())
        .item()
    )
    outfile = Path(outpath, "imputed_" + filename + ".bed")
    print(f"Saving {filename}")
    df.write_csv(outfile, separator="\t")
    return f"Saved {filename}"


def parallel_save(dfs, outpath):
    """Save files in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(save_files, df, outpath): df for df in dfs}

    for future in concurrent.futures.as_completed(futures):
        df = futures[future]
        result = future.result()
        print(f"{result}")
    return df


# def save_files_normal(file, outpath):
#     """Save files w/o streaming."""
#     filename = (
#         file.unique(subset="sample", keep="any")
#         .select(pl.col("sample").filter(pl.col("sample") != "imputed").first())
#         .item()
#     )
#     outfile = Path(outpath, "imputed_" + filename + ".bed")
#     print(f"Saving {filename}")
#     file.write_csv(outfile, separator="\t")
#     return f"Saved {filename}"


# def save_files_streaming(file, outpath):
#     """Save files by streaming."""
#     # file = file.collect(streaming=True)
#     filename = (
#         file.unique(subset="sample", keep="any")
#         .select(pl.col("sample").filter(pl.col("sample") != "imputed").first())
#         .item()
#     )
#     outfile = Path(outpath, "imputed_" + filename + ".bed")
#     print(f"Saving {filename}")
#     file.write_csv(outfile, separator="\t")
#     return f"Saved {filename}"
