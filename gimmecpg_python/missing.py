"""Identify missing sites."""

import polars as pl


def missing_sites(bed, ref):
    """Compare to reference."""
    ref = (
        pl.scan_parquet(ref, parallel="row_groups")
        .cast({"chr": pl.Utf8, "start": pl.UInt64, "end": pl.UInt64})
        .select(["chr", "start", "end"])
    )

    missing = ref.join(bed, on=["chr", "start"], how="left")
    print(missing.filter(pl.col("avg").is_not_null()).collect())
    quit()

    missing_mat = (
        missing.with_columns(
            pl.when(pl.col("avg").is_not_null()).then(pl.col("start")).alias("b_start"),
            pl.when(pl.col("avg").is_not_null()).then(pl.col("start")).alias("f_start"),
            pl.when(pl.col("avg").is_not_null()).then(pl.col("avg")).alias("b_meth"),
            pl.when(pl.col("avg").is_not_null()).then(pl.col("avg")).alias("f_meth"),
        )
        .with_columns(
            pl.col(["f_start", "f_meth"]).backward_fill().over("chr"),
            pl.col(["b_start", "b_meth"]).forward_fill().over("chr"),
        )
        .with_columns(
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )
    )

    return missing_mat
