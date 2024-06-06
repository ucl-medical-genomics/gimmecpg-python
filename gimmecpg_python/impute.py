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


def h2oPrep(lf, dist, streaming):
    """Prepare training and testing frames."""
    known_lf = (
        lf.filter(pl.col("avg").is_not_null())
        .with_columns(
            pl.col("start").shift(-1).over("chr").alias("f_start"),
            pl.col("start").shift().over("chr").alias("b_start"),
        )
        .drop_nulls()
        .with_columns(
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )
        .select(["avg", "b_meth", "f_meth", "b_dist", "f_dist"])
    )

    to_predict_lf = lf.filter(pl.col("avg").is_null()).drop(["end_right", "b_start", "f_start"])

    if dist is not None:
        to_predict_lf = to_predict_lf.filter((pl.col("f_dist") < dist) & (pl.col("b_dist") < dist))

    if streaming:
        known = known_lf.collect(streaming=True)
        to_predict = to_predict_lf.collect(streaming=True)
    else:
        known = known_lf.collect()
        to_predict = to_predict_lf.collect()

    return known, to_predict, to_predict_lf


def h2oTraining(lf, maxTime, maxModels, dist, streaming):  
    """Do training."""
    print("Starting H2O AutoML training")

    training, test, to_predict_lf = h2oPrep(lf, dist, streaming)

    h2o.init()

    trainingFrame = h2o.H2OFrame(training.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    trainingFrame[["avg", "b_meth", "f_meth", "b_dist", "f_dist"]] = trainingFrame[
        ["avg", "b_meth", "f_meth", "b_dist", "f_dist"]
    ].asnumeric()

    testingFrame = h2o.H2OFrame(test.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    testingFrame[["avg", "b_meth", "f_meth", "b_dist", "f_dist"]] = testingFrame[
        ["avg", "b_meth", "f_meth", "b_dist", "f_dist"]
    ].asnumeric()

    y = "avg"  # specify the response columns
    x = ["b_meth", "f_meth", "b_dist", "f_dist"]  # specify the predictors

    aml = H2OAutoML(max_runtime_secs=maxTime, max_models=maxModels, seed=1)
    aml.train(y=y, x=x, training_frame=trainingFrame)  # training_frame = train, leaderboard_frame = test
    lb = aml.leaderboard

    prediction = aml.leader.predict(testingFrame)
    prediction_lf = pl.LazyFrame(prediction.as_data_frame())

    imputed_lf = pl.concat([to_predict_lf, prediction_lf], how="horizontal")

    if dist is not None:
        lf = lf.filter((pl.col("f_dist") < dist) & (pl.col("b_dist") < dist))

    res = (
        lf.join(imputed_lf, on=["chr", "start", "end"], how="full", coalesce = True)
        .with_columns(pl.col("avg").fill_null(pl.col("predict")), pl.col("sample").fill_null(pl.lit("imputed")))
        .with_columns(
            avg=pl.when(pl.col("avg") > 100).then(100).when(pl.col("avg") < 0).then(0).otherwise(pl.col("avg"))
        )
    )

    res = res.select(["chr", "start", "end", "strand", "sample", "avg"])

    lb.head(rows=lb.nrows)
    return res
