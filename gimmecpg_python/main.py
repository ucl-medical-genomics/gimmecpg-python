"""Main module."""

import argparse
import glob
import sys

import polars as pl
from itertools import batched
from files import read_files, save_files_normal, save_files_streaming
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
    required=False,
    type=int,
    help="Maximum distance between missing site and each neighbour for the site to be imputed. \
                       Default = all sites considered",
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
    "-a",
    "--accurate",
    action="store_true",
    required=False,
    help="Choose between Accurate and Fast mode. Default = Fast",
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
    default=5,
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
if args.pattern:
    bed_files = args.input + "/*" + args.pattern + "*.bed"
else:
    bed_files = args.input + "/*.bed"


bed_paths = glob.glob(bed_files)

# check files exist
if not bed_paths:
    print("ERROR: No matching Bed file(s) found. GIMMEcpg terminating.")
    sys.exit(1)


print(f"Merge methylation sites on opposite strands = {args.collapse}")

lf_list = [read_files(bed, args.minCov, args.collapse) for bed in bed_paths]


##########################
# Identify missing sites #
##########################

missing = [missing_sites(lf, args.ref) for lf in lf_list]
print("Identified missing sites")

################################
# Imputation (default is fast) #
################################

if args.maxDistance is not None:
    print(f"Imputing methylation for missing sites less than {args.maxDistance} bases from each neighbour")

results = []

if not args.accurate:
    print("Fast imputation mode")
    imputed_lfs = [fast_impute(lf, args.maxDistance) for lf in missing]  # RESULT
    results = imputed_lfs
else:
    print("Accurate mode: prepare for H2O AutoML training")
    lead_prediction = [
        h2oTraining(lf, args.runTime, args.maxModels, args.maxDistance, args.streaming) for lf in missing
    ]  # RESULT
    results = lead_prediction


if args.streaming:
        print("Collecting fast imputation results in streaming mode")
        for sample in results:
            save_files_streaming(sample, args.output)
        print("All files Saved")
else:
        print("Collecting fast imputation results")
        if len(results) <= 10:
            dfs = pl.collect_all(results)
            for sample in dfs:
                save_files_normal(sample, args.output)
            print("All files Saved")
        else:
            for batch in batched(results, 10):
                dfs = pl.collect_all(batch)
                print("Saving in batches")
                for sample in dfs:
                    save_files_normal(sample, args.output)
            print("All files Saved")

print("Imputation complete")
