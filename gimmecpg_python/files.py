"""Input and output files."""

from pathlib import Path

import polars as pl


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
                "column_10": pl.UInt16,
                "column_11": pl.UInt16,
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
        .filter(pl.col("coverage") > mincov)  # filter by coverage first
    )

    if collapse:
        pos = bed.filter(pl.col("strand") == "+").with_columns(
            (pl.col("end")).alias("reverse_start")
        )  # add column for start site on complementary strand
        neg = bed.filter(pl.col("strand") == "-")

        # wCompStart = bed.with_columns((pl.col('end') + 1)
        # .alias('reverse_start')) # if data is 1-index Start side

        joint = neg.join(
            pos, left_on=["chr", "start"], right_on=["chr", "reverse_start"], how="outer_coalesce"
        ).with_columns(pl.concat_str([pl.col("strand"), pl.col("strand_right")], separator="/", ignore_nulls=True))

        results = (
            joint.with_columns(
                pl.min_horizontal("start", "start_right").alias("start"),
                pl.max_horizontal("end", "end_right").alias("end"),
                pl.col(["percent_methylated_right", "coverage_right", "percent_methylated", "coverage"])
                .fill_null(0)
                .cast(pl.UInt16),
            )
            .with_columns(
                (
                    (
                        pl.col("coverage") * pl.col("percent_methylated")
                        + pl.col("coverage_right") * pl.col("percent_methylated_right")
                    )
                    / (pl.col("coverage") + pl.col("coverage_right"))
                ).alias("avg")
            )  # calculated weighted average
            .select(["chr", "start", "end", "strand", "avg"])
            .with_columns(pl.lit(name).alias("sample"))
        )
    else:
        results = (
            bed.with_columns(pl.col("coverage").alias("avg"))
            .select(["chr", "start", "end", "strand", "avg"])
            .with_columns(pl.lit(name).alias("sample"))
        )

    return results


def save_files_normal(file, outpath):
    """Save files w/o streaming."""
    filename = (
        file.unique(subset="sample", keep="any")
        .select(pl.col("sample").filter(pl.col("sample") != "imputed").first())
        .item()
    )
    outfile = Path(outpath, "imputed_" + filename + ".bed")
    file.write_csv(outfile, separator="\t")


def save_files_streaming(file, outpath):
    """Save files by streaming."""
    filename = (
        file.unique(subset="sample", keep="any")
        .select(pl.col("sample").filter(pl.col("sample") != "imputed").first())
        .item()
    )
    outfile = Path(outpath, "imputed_" + filename + ".bed")
    file.write_csv(outfile, separator="\t")
