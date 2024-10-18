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
        pl.col("sample").fill_null(pl.lit("imputed"))
    )

    imputed = imputed.select(["chr", "start", "end", "strand", "sample", "avg"])

    return imputed


def distBins(lf, col):
    """Bin neighbouring distances."""
    corrMat = (
        lf.with_columns(
            pl.when(pl.col(col) <= 200)
                .then(pl.col(col))
            .alias("dist_bins")
        )
        .with_columns(
            pl.when(pl.col(col) > 200)  
                .then((((pl.col(col)-1)//10) + 1) * 10)
                .otherwise(pl.col("dist_bins"))
            .alias("dist_bins")
        )
        .with_columns(
            pl.when(pl.col(col) > 500)  
                .then((((pl.col(col)-1)//100) + 1) * 100)
                .otherwise(pl.col("dist_bins"))
            .alias("dist_bins")
        )
        .with_columns(
            pl.when(pl.col(col) > 2000)  
                .then(-1)
                .otherwise(pl.col("dist_bins"))
            .alias("dist_bins")
        )
    )
    
    return corrMat


def h2oPrep(lf, dist, streaming):
    """Prepare training and testing frames."""

    known_sites = (
        lf.filter(pl.col("avg").is_not_null())
        .select(["chr", "start", "end", "avg"])
        .with_columns(
            pl.col("start").shift(-1).over("chr").alias("f_start"),
            pl.col("start").shift().over("chr").alias("b_start"),
            pl.col("avg").shift(-1).over("chr").alias("f_meth"),
            pl.col("avg").shift().over("chr").alias("b_meth")
        )
        .with_columns(
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )        
        .with_columns(
            ((pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist")) / pl.sum_horizontal("f_dist", "b_dist")).alias("fast_res")
        )
        .with_columns(
            (pl.col("fast_res") - pl.col("avg")).abs().alias("error")
        )
        .with_columns(
            (pl.col("error").quantile(0.99, "nearest")).alias("limit") # 0.99 is the sweet spot?
        )
        .filter(pl.col("error") < pl.col("limit"))
    )


    if dist > 0:
        known_sites = known_sites.filter((pl.col("f_dist") <= dist), (pl.col("b_dist") <= dist))


    known_sites = distBins(known_sites, "f_dist").rename({"dist_bins": "f_dist_bins"})
    known_sites = distBins(known_sites, "b_dist").rename({"dist_bins": "b_dist_bins"})

    corr = (known_sites.select(["avg", "f_meth", "f_dist", "f_dist_bins"])
            .group_by("f_dist_bins").agg(pl.corr("f_meth","avg", method = "pearson").alias("corr"))
            .with_columns(
                pl.when(pl.col("f_dist_bins") == -1).then(0).otherwise(pl.col("corr")).alias("corr")
            )
    ).drop_nulls()


    features_lf = (
        known_sites
        .join(corr, right_on = "f_dist_bins", left_on = "b_dist_bins", how = "left").rename({"corr": "b_corr"})
        .join(corr, on = "f_dist_bins", how = "left").rename({"corr": "f_corr"})
        .select(["avg", "b_meth", "f_meth", "b_dist", "f_dist", "b_corr", "f_corr"])
        .with_columns(
            (pl.col("b_meth").log1p()).alias("lg_b_meth"),
            (pl.col("f_meth").log1p()).alias("lg_f_meth"),
            (pl.col("b_dist").log1p()).alias("lg_b_dist"),
            (pl.col("f_dist").log1p()).alias("lg_f_dist")
        )
    )
    
    missing_sites = lf.filter(pl.col("avg").is_null())

    missing_sites = distBins(missing_sites, "f_dist").rename({"dist_bins": "f_dist_bins"})
    missing_sites = distBins(missing_sites, "b_dist").rename({"dist_bins": "b_dist_bins"})

    to_predict_lf = (
        missing_sites
        .join(corr, right_on = "f_dist_bins", left_on = "b_dist_bins", how = "left").rename({"corr": "b_corr"})
        .join(corr, on = "f_dist_bins", how = "left").rename({"corr": "f_corr"})
        .with_columns(
            (pl.col("b_meth").log1p()).alias("lg_b_meth"),
            (pl.col("f_meth").log1p()).alias("lg_f_meth"),
            (pl.col("b_dist").log1p()).alias("lg_b_dist"),
            (pl.col("f_dist").log1p()).alias("lg_f_dist")
        )
    )

    if streaming:
        features = features_lf.collect(streaming=True)
        to_predict = to_predict_lf.select(["avg", "lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"]).collect(streaming=True)
    else:
        features = features_lf.collect()
        to_predict = to_predict_lf.select(["avg", "lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"]).collect()

    return features, to_predict, to_predict_lf


def h2oTraining(lf, maxTime, maxModels, dist, streaming):
    """Do training."""
    print("Starting H2O AutoML training")

    training, test, to_predict_lf = h2oPrep(lf, dist, streaming)

    h2o.init()

    trainingFrame = h2o.H2OFrame(
        training.to_pandas(use_pyarrow_extension_array=True)
    )  # make sure it's the right format
    trainingFrame[["avg", "lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"]] = trainingFrame[
        ["avg", "lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"] 
    ].asnumeric()

    testingFrame = h2o.H2OFrame(test.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    testingFrame[["avg", "lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"]] = testingFrame[
        ["avg", "lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"] 
    ].asnumeric()

    y = "avg"  # specify the response columns
    x = ["lg_b_dist", "lg_f_dist", "lg_b_meth", "lg_f_meth", "b_corr", "f_corr"]  # specify the predictors 

    aml = H2OAutoML(max_runtime_secs=maxTime, seed=1, max_models=maxModels, nfolds = 5, stopping_rounds = 3, sort_metric = "deviance") 
    aml.train(y=y, x=x, training_frame=trainingFrame)  
    lb = aml.leaderboard

    prediction = aml.leader.predict(testingFrame)

    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
        prediction_df = prediction.as_data_frame()

    prediction_lf = pl.LazyFrame(prediction_df)

    imputed_lf = pl.concat([to_predict_lf, prediction_lf], how="horizontal")

    res = (
        lf.join(imputed_lf, on=["chr", "start"], how="full", coalesce=True)
        .with_columns(pl.col("avg").fill_null(pl.col("predict")), pl.col("sample").fill_null(pl.lit("imputed")))
        .with_columns(
            avg=pl.col("avg").clip(0, 100)
        )
    )

    if dist > 0:
        res = res.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))

    res = res.select(["chr", "start", "end", "strand", "sample", "avg"])
    
    print(lb.head(rows=lb.nrows))

    h2o.remove_all()

    return res
