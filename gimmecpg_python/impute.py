"""Imputation."""

import h2o
import polars as pl
from h2o.automl import H2OAutoML


def fast_impute(lf, dist):
    """Fast imputation."""

    lf = lf.drop("methylation") # remove the old methylation column

    if dist > 0:
        lf = lf.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))

    neighbours_added = (
        lf.with_columns(
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
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), 
            (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )
    )

    imputed = (neighbours_added.with_columns(
            sample = pl.when(pl.col("avg").is_null()).then(pl.lit("imputed")).otherwise("sample")
        )    
        .with_columns(
            pl.col("avg").fill_null(
                (pl.col("b_meth") * pl.col("f_dist") + pl.col("f_meth") * pl.col("b_dist"))
                / pl.sum_horizontal("f_dist", "b_dist")
            )
        )
        .with_columns(
            methylation = pl.col("avg")
        )
    )

    imputed = imputed.select(["chr", "start", "end", "strand", "sample", "methylation", "total_coverage"])

    return imputed


def h2oPrep(lf, dist, streaming):
    """Prepare training and testing frames."""

    features_lf = (
        lf.filter(pl.col("avg").is_not_null()) # any CpG site that is present in the original data, regardless of coverage, is considered "known". "Methylation" column will only be null if CpG site is in reference but not in original data
        .with_columns(
            pl.col("start").shift(-1).over("chr").alias("f_start"),
            pl.col("start").shift().over("chr").alias("b_start"),
            pl.col("methylation").shift(-1).over("chr").alias("f_meth"),
            pl.col("methylation").shift().over("chr").alias("b_meth")
        )
        .with_columns(
            (pl.col("start") - pl.col("b_start")).alias("b_dist"), 
            (pl.col("f_start") - pl.col("start")).alias("f_dist")
        )        
        
    )

    to_predict_lf = lf.filter(pl.col("avg").is_null()) # "avg" is null when either site from reference not present in sample data, or if coverage < threshold

    if dist > 0:
        to_predict_lf = to_predict_lf.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))
        features_lf = features_lf.filter((pl.col("f_dist") <= dist) & (pl.col("b_dist") <= dist))


    if streaming:
        features = features_lf.select(["methylation", "b_dist", "f_dist", "b_meth", "f_meth"]).collect(streaming=True)
        to_predict = to_predict_lf.select(["methylation", "b_dist", "f_dist", "b_meth", "f_meth"]).collect(streaming=True)
    else:
        features = features_lf.select(["methylation", "b_dist", "f_dist", "b_meth", "f_meth"]).collect()
        to_predict = to_predict_lf.select(["methylation", "b_dist", "f_dist", "b_meth", "f_meth"]).collect()

    return features, to_predict, to_predict_lf


def h2oTraining(lf, maxTime, maxModels, dist, streaming):
    """Do training."""
    print("Starting H2O AutoML training")

    training, test, to_predict_lf = h2oPrep(lf, dist, streaming)

    h2o.init(port = 54321, nthreads = 24, max_mem_size = "100G")

    trainingFrame = h2o.H2OFrame(
        training.to_pandas(use_pyarrow_extension_array=True)
    )  # make sure it's the right format
    trainingFrame[["methylation", "b_dist", "f_dist", "b_meth", "f_meth"]] = trainingFrame[
        ["methylation", "b_dist", "f_dist", "b_meth", "f_meth"] 
    ].asnumeric()

    testingFrame = h2o.H2OFrame(test.to_pandas(use_pyarrow_extension_array=True))  # make sure it's the right format
    testingFrame[["methylation", "b_dist", "f_dist", "b_meth", "f_meth"]] = testingFrame[
        ["methylation", "b_dist", "f_dist", "b_meth", "f_meth"] 
    ].asnumeric()

    y = "methylation"  # specify the response columns
    x = ["b_dist", "f_dist", "b_meth", "f_meth"]  # specify the predictors 

    aml = H2OAutoML(max_runtime_secs=maxTime, max_models = maxModels, seed=1234, sort_metric = "MAE") 
    aml.train(y=y, x=x, training_frame=trainingFrame)  
    lb = aml.leaderboard

    prediction = aml.leader.predict(testingFrame)

    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
        prediction_df = prediction.as_data_frame() # results

    prediction_lf = pl.LazyFrame(prediction_df)

    imputed_lf = pl.concat([to_predict_lf, prediction_lf], how="horizontal")

    res = (
        lf.join(imputed_lf, on=["chr", "start"], how="full", coalesce=True)
        .with_columns(pl.col("avg").fill_null(pl.col("predict")), pl.col("sample").fill_null(pl.lit("imputed")))
        .with_columns(
             methylation = pl.col("avg").clip(0, 100)
        )
    )

    res = res.select(["chr", "start", "end", "strand", "sample", "methylation", "total_coverage"])
    
    print(lb.head(rows=lb.nrows))

    h2o.remove_all()

    return res
