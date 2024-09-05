"""Main module."""

import argparse
import glob
import re
import sys
from itertools import batched

import polars as pl
from files import parallel_save, read_files
from impute import fast_impute, h2oTraining
from missing import missing_sites

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser(description="Options for imputing missing CpG sites based on neighbouring sites")
parser.add_argument(
    "-i",
    "--input",
    action="store",
    required=True,
    help="Path to directory of bed files",
)
parser.add_argument("-p", "--pattern", action="store", required=False, help="Pattern to select specific files")
parser.add_argument("-e", "--exclude", action="store", required=False, help="Path to a list of CpG sites to exclude")
parser.add_argument("-o", "--output", action="store", required=True, help="Path to output directory")
parser.add_argument("-r", "--ref", action="store", required=True, help="Path to reference methylation file")
parser.add_argument(
    "-c",
    "--minCov",
    action="store",
    default=10,
    required=False,
    type=int,
    help="Minimum coverage to consider methylation site as present. Default = 10",
)
parser.add_argument(
    "-d",
    "--maxDistance",
    action="store",
    default=1000,
    required=False,
    type=int,
    help="Maximum distance between missing site and each neighbour for the site to be imputed. \
                       Default = 1000bp",
)
parser.add_argument(
    "-k",
    "--collapse",
    action="store_false",
    required=False,
    help="Choose whether to merge methylation sites on opposite \
                       strands together. Default = True",
)
parser.add_argument(
    "-x",
    "--machineLearning",
    action="store_true",
    required=False,
    help="Choose whether to use machine learning for imputation. Default = no machine learning",
)
parser.add_argument(
    "-t",
    "--runTime",
    action="store",
    default=3600,
    required=False,
    type=int,
    help="Time (seconds) to train model. Default = 3600s (2h)",
)
parser.add_argument(
    "-m",
    "--maxModels",
    action="store",
    # default=5,
    required=False,
    type=int,
    help="Maximum number of models to train within the time specified \
                     under --runTime. Excludes Stacked Ensemble models",
)
parser.add_argument(
    "-s",
    "--streaming",
    action="store_true",
    required=False,
    help="Choose if streaming is required (for files that exceed memory). Default = False",
)
args = parser.parse_args()


##########################################################
# Read file in as LazyFrame, collapse strands if needed. #
##########################################################

# select files
bed_files = args.input + "/*.bed"
bed_paths = glob.glob(bed_files)

if args.pattern:
    names = args.pattern.split(",")
    regex = re.compile("|".join(names))
    bed_paths = [path for path in bed_paths if regex.search(path)]


# check files exist
if not bed_paths:
    print("ERROR: No matching Bed file(s) found. GIMMEcpg terminating.")
    sys.exit(1)


print(f"Merge methylation sites on opposite strands = {args.collapse}")
print(f"Coverage cutoff at {args.minCov}")

lf_list = [read_files(bed, args.minCov, args.collapse) for bed in bed_paths]


##########################
# Identify missing sites #
##########################

missing = [missing_sites(lf, args.ref, args.exclude) for lf in lf_list]
print("Identified missing sites")

################################
# Imputation (default is fast) #
################################

if args.maxDistance > 0:
    print(f"Imputing methylation for missing sites within {args.maxDistance} bases from each neighbour")

results = []

if not args.machineLearning:
    print("Default imputation mode")
    imputed_lfs = [fast_impute(lf, args.maxDistance) for lf in missing]  # RESULT
    results = imputed_lfs
else:
    print("machineLearning mode: prepare for H2O AutoML training")
    lead_prediction = [
        h2oTraining(lf, args.runTime, args.maxModels, args.maxDistance, args.streaming) for lf in missing
    ]  # RESULT
    results = lead_prediction


# if args.streaming:
#         print("Collecting fast imputation results in streaming mode")
#         dfs = pl.collect_all(results, streaming = True)
#         for sample in dfs:
#             save_files_streaming(sample, args.output)
#         print("Files Saved")
# else:
#         print("Collecting fast imputation results")
#         dfs = pl.collect_all(results)
#         for sample in dfs:
#             save_files_normal(sample, args.output)
#         print("Files Saved")

batch_limit = 10

if len(results) <= batch_limit:
    print("Batch mode OFF")
    if args.streaming:
        print("Collecting results in streaming mode")
        dfs = pl.collect_all(results, streaming=True)
        parallel_save(dfs, args.output)
        print("All files Saved")
    else:
        print("Collecting results")
        dfs = pl.collect_all(results)
        parallel_save(dfs, args.output)
        print("All files Saved")
else:
    print(f"Batches of {batch_limit}")
    if args.streaming:
        print("Collecting batches of results in streaming mode")
        for batch in batched(results, batch_limit):
            dfs = pl.collect_all(batch, streaming=True)
            print("Saving in batches")
            parallel_save(dfs, args.output)
        print("All files Saved")
    else:
        print("Collecting batches of results")
        for batch in batched(results, batch_limit):
            dfs = pl.collect_all(batch)
            print("Saving in batches")
            parallel_save(dfs, args.output)
        print("All files Saved")


print("Imputation complete")
