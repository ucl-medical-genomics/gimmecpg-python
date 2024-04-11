"""Imputation."""

import h2o
import polars as pl
from h2o.automl import H2OAutoML


def fast_impute(lf, dist):
    """Fast imputation."""
    if dist is not None:
        lf = lf.filter((pl.col("f_dist") < dist) & (pl.col("b_dist") < dist))

    imputed = lf.with_columns(
        pl.col("avg").fill_null(
            (pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist"))
            / pl.sum_horizontal("f_dist", "b_dist")
        ),
        pl.col("sample").fill_null(pl.lit("imputed")),
    )

    imputed = imputed.select(["chr", "start", "end", "strand", "sample", "avg"])

    return imputed


def h2oTraining(lf, maxTime, maxModels, dist, streaming):
    """Accurate imputation."""
    full = (
        lf.filter(pl.col("avg").is_not_null())
        .with_columns(
            pl.col("start").shift(-1).over("chr").alias("f_start"),
            pl.col("start").shift().over("chr").alias("b_start"),
        )
        .drop_nulls()
        .with_columns(
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )
    )

    to_predict_lf = lf.filter(pl.col("avg").is_null())

    if dist is not None:
        print(f"Imputing methylation for missing sites less than {dist} bases from each neighbour")
        to_predict_lf = to_predict_lf.filter((pl.col("f_dist") < dist) & (pl.col("b_dist") < dist))

    if streaming:
        full = full.collect(streaming=True)
        to_predict = to_predict_lf.collect(streaming=True)
    else:
        full = full.collect()
        to_predict = to_predict_lf.collect()

    print("Starting H2O AutoML training")
    h2o.init()

    full = h2o.H2OFrame(full.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    full[["avg", "b_meth", "f_meth", "b_dist", "f_dist"]] = full[
        ["avg", "b_meth", "f_meth", "b_dist", "f_dist"]
    ].asnumeric()

    to_predict = h2o.H2OFrame(to_predict.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    to_predict[["avg", "b_meth", "f_meth", "b_dist", "f_dist"]] = to_predict[
        ["avg", "b_meth", "f_meth", "b_dist", "f_dist"]
    ].asnumeric()

    y = "avg"  # specify the response columns
    x = ["b_meth", "f_meth", "b_dist", "f_dist"]  # specify the predictors

    aml = H2OAutoML(max_runtime_secs=maxTime, max_models=maxModels, seed=1)
    aml.train(y=y, x=x, training_frame=full)  # training_frame = train, leaderboard_frame = test
    lb = aml.leaderboard

    prediction = aml.leader.predict(to_predict)
    prediction_lf = pl.LazyFrame(prediction.as_data_frame())

    imputed_lf = pl.concat([to_predict_lf, prediction_lf], how="horizontal")

    if dist is not None:
        lf = lf.filter((pl.col("f_dist") < dist) & (pl.col("b_dist") < dist))

    res = (
        lf.join(imputed_lf, on=["chr", "start", "end"], how="outer_coalesce")
        .with_columns(pl.col("avg").fill_null(pl.col("predict")), pl.col("sample").fill_null(pl.lit("imputed")))
        .with_columns(
            avg=pl.when(pl.col("avg") > 100).then(100).when(pl.col("avg") < 0).then(0).otherwise(pl.col("avg"))
        )
    )

    res = res.select(["chr", "start", "end", "strand", "sample", "avg"])

    print(lb)
    return res
