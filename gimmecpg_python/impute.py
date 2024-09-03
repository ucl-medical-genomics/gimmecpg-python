"""Imputation."""

import h2o
import polars as pl
from h2o.automl import H2OAutoML


def fast_impute(lf, dist):
    """Fast imputation."""
    if dist > 0:
        lf = lf.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))

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
    
    known_sites = (
        lf.filter(pl.col("avg").is_not_null())
        .select(["chr", "start", "end", "avg"])
    )

    features_lf = (
        known_sites.with_columns(
            pl.col("start").shift(-1).over("chr").alias("f_start"),
            pl.col("start").shift().over("chr").alias("b_start"),
            pl.col("avg").shift(-1).over("chr").alias("f_meth"),
            pl.col("avg").shift().over("chr").alias("b_meth")
        )
        .drop_nulls()
        .with_columns(
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )
        .with_columns(
            pl.sum_horizontal("f_dist", "b_dist").alias("t_dist")
        )
        .with_columns(
            ((pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist")) / pl.col("t_dist")).alias("fast_res"),
            pl.col("t_dist").quantile(0.999, "nearest").alias("limit")

        )
        .filter(pl.col("t_dist") < pl.col("limit"))
        .select(["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"])
    )

    # with pl.Config(tbl_cols = -1):
    #     print(features_lf.fetch(10000))
    # exit()
    
    # features = ( known_lf.with_columns(
    #         pl.col("start").shift(-1).over("chr").alias("f_start"),
    #         pl.col("start").shift().over("chr").alias("b_start"),
    #         pl.col("avg").shift(-1).over("chr").alias("f_meth"),
    #         pl.col("avg").shift().over("chr").alias("b_meth")
    #     )
    #     .drop_nulls()
    #     .with_columns(
    #         (pl.col("start") - pl.col("b_start")).alias("b_dist"), (pl.col("f_start") - pl.col("start")).alias("f_dist")
    #     )
    #     .with_columns(
    #         t_dist = pl.sum_horizontal("f_dist", "b_dist")
    #     )
    #     # .with_columns(
    #     #     avgMeth = (pl.col("b_meth") + pl.col("f_meth"))/2
    #     # )
    #     .with_columns(
    #         ((pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist")) / pl.sum_horizontal("f_dist", "b_dist")).alias("fast_res"),
    #         # (pl.col("b_meth") * pl.col("f_dist")).log1p().alias("bMeth_fDist"),
    #         # (pl.col("f_meth") * pl.col("b_dist")).log1p().alias("fMeth_bDist"),
    #         # (pl.col("b_dist") + pl.col("f_dist")).alias("total_distance"),
    #         # (pl.col("avgMeth").mode()).alias("modeVal"),
    #         # (pl.col("b_meth").log1p()).alias("lg_b_meth"),
    #         # (pl.col("f_meth").log1p()).alias("lg_f_meth"),
    #         # (pl.col("b_dist").log1p()).alias("lg_b_dist"),
    #         # (pl.col("f_dist").log1p()).alias("lg_f_dist"),
    #         # (pl.col("b_meth") + pl.col("f_meth")).alias("total_meth"),
    #         pl.col("avg").qcut(5, labels=["a", "b", "c", "d", "e"]).alias("bins"), # ??
    #         limit = pl.col("t_dist").quantile(0.999, "nearest")
    #         # weights = (1/((pl.col("b_dist") + pl.col("f_dist")).log(base=10)))
    #     )
    #     .filter(pl.col("t_dist") < pl.col("limit"))
    #     .with_columns(
    #         (pl.int_range(pl.count("bins"), eager=False).over("bins") % 5).alias("folds")
    #     )
    #     .select(["avg", "b_meth", "f_meth", "b_dist", "f_dist", "folds", "fast_res"])
    # )
    
    to_predict_lf = lf.filter(pl.col("avg").is_null()).with_columns(
            ((pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist")) / pl.sum_horizontal("f_dist", "b_dist")).alias("fast_res")
        )

    # to_predict_lf = lf.filter(pl.col("avg").is_null()).with_columns(
    #         # ((pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist")) / pl.sum_horizontal("f_dist", "b_dist")).alias("fast_res"),
    #         # (pl.col("b_meth") * pl.col("f_dist")).log1p().alias("bMeth_fDist"),
    #         # (pl.col("f_meth") * pl.col("b_dist")).log1p().alias("fMeth_bDist"),
    #         # (pl.col("b_meth").log1p()).alias("lg_b_meth"),
    #         # (pl.col("f_meth").log1p()).alias("lg_f_meth"),
    #         # (pl.col("b_dist").log1p()).alias("lg_b_dist"),
    #         # (pl.col("f_dist").log1p()).alias("lg_f_dist"),
    #         # pl.col("avg").cast(pl.UInt32),
    #         # pl.col("b_meth").cast(pl.UInt32),
    #         # pl.col("f_meth").cast(pl.UInt32),
    #         # pl.col("b_dist").cast(pl.UInt32),
    #         # pl.col("f_dist").cast(pl.UInt32),
    #         # weights = (1/((pl.col("b_dist") + pl.col("f_dist")).log(base=10)))
    #         # (pl.col("b_dist") + pl.col("f_dist")).alias("total_distance")
    #         # ((pl.col("b_dist") + pl.col("f_dist"))/2).alias("total_distance")
    #     ).drop(["end_right", "b_start", "f_start"])

    # if dist > 0:
    #     to_predict_lf = to_predict_lf.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))
    # with pl.Config(tbl_cols = -1):
    #     print(to_predict_lf.select(["avg", "b_meth", "f_meth", "b_dist", "f_dist"]).fetch(1000))
    # exit()

    if streaming:
        features = features_lf.collect(streaming=True)
        to_predict = to_predict_lf.select(["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"]).collect(streaming=True)
    else:
        features = features_lf.collect()
        to_predict = to_predict_lf.select(["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"]).collect()

    return features, to_predict, to_predict_lf


def h2oTraining(lf, maxTime, maxModels, dist, streaming):
    """Do training."""
    print("Starting H2O AutoML training")

    training, test, to_predict_lf = h2oPrep(lf, dist, streaming)

    h2o.init()

    trainingFrame = h2o.H2OFrame(
        training.to_pandas(use_pyarrow_extension_array=True)
    )  # make sure it's the right format
    trainingFrame[["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"]] = trainingFrame[
        ["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"] #, "total_distance", "weights" , "lg_b_dist", "lg_f_dist" , "b_dist", "f_dist", "weights"
    ].asnumeric()

    testingFrame = h2o.H2OFrame(test.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    testingFrame[["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"]] = testingFrame[
        ["avg", "b_meth", "f_meth", "b_dist", "f_dist", "fast_res"] # , "total_distance", "weights" , "lg_b_dist", "lg_f_dist"
    ].asnumeric()

    y = "avg"  # specify the response columns
    x = ["b_meth", "f_meth", "b_dist", "f_dist", "fast_res"]  # specify the predictors , "weights" , "total_distance", , "lg_b_dist", "lg_f_dist" , "fast_res"

    aml = H2OAutoML(max_runtime_secs=maxTime, seed=1, max_models=maxModels) # , stopping_metric = "MAE", sort_metric = "MAE" , distribution = "poisson" 
    aml.train(y=y, x=x, training_frame=trainingFrame)  # training_frame = train, leaderboard_frame = testingFrame, , weights_column = "weights" , fold_column="folds"
    lb = aml.leaderboard

    prediction = aml.leader.predict(testingFrame)

    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
        prediction_df = prediction.as_data_frame()

    prediction_lf = pl.LazyFrame(prediction_df)

    imputed_lf = pl.concat([to_predict_lf, prediction_lf], how="horizontal")

    # if dist is not None:
    #     lf = lf.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))

    res = (
        lf.join(imputed_lf, on=["chr", "start", "end"], how="full", coalesce=True)
        .with_columns(pl.col("avg").fill_null(pl.col("predict")), pl.col("sample").fill_null(pl.lit("imputed")))
        .with_columns(
            avg=pl.col("avg").clip(0, 100)
        )
    )

    if dist > 0:
        res = res.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))

    res = res.select(["chr", "start", "end", "strand", "sample", "avg"])
    
    # with pl.Config(tbl_cols = -1):
    #     print(res.fetch(1000))
    # exit()

    print(lb.head(rows=lb.nrows))

    h2o.remove_all()

    return res
