# GIMMEcpg-python


[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=plastic)](https://github.com/ucl-medical-genomics/gimmecpg.py/graphs/commit-activity)
[![GitHub](https://img.shields.io/github/license/ucl-medical-genomics/gimmecpg.py?style=plastic)](https://github.com/ucl-medical-genomics/gimmecpg.py)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/ucl-medical-genomics/gimmecpg.py?display_name=tag&logo=github&style=plastic)](https://github.com/ucl-medical-genomics/gimmecpg.py)
[![GitHub Release](https://img.shields.io/github/release-date/ucl-medical-genomics/gimmecpg.py?style=plastic&logo=github)](https://github.com/ucl-medical-genomics/gimmecpg.py)
[![Poetry](https://img.shields.io/endpoint?style=plastic&url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?style=plastic&url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=plastic)](https://github.com/pre-commit/pre-commit)

## About The Project

Python version of GIMMEcpg, developed with Polars and H2OAutoML

## Getting Started

```
usage: main.py [-h] -i INPUT -o OUTPUT -r REF [-c MINCOV] [-d MAXDISTANCE]
[-k] [-a] [-t RUNTIME] [-m MAXMODELS] [-s]

Options for imputing missing CpG sites based on neighbouring sites:

-h, --help           show this help message and exit
-i, --input          Path to directory of bed files (make sure it contains only the bed files to be analysed)
-o, --output         Path to output directory
-r, --ref            Path to reference methylation file
-c, --minCov         Minimum coverage to consider methylation site as present. Default = 10
-d, --maxDistance    Maximum distance between missing site and each neighbour for the site to be imputed. Default = all sites considered
-k, --collapse       Choose whether to merge methylation sites on opposite strands together. Default = False
-a, --accurate       Choose between Accurate and Fast mode. Default = Fast
-t, --runTime        Time (seconds) to train model. Default = 3600s (2h)
-m, --maxModels      Maximum number of models to train within the time specified under --runTime. Excludes Stacked Ensemble models
-s, --streaming      Choose if streaming is required (for files that exceed memory). Default = False
```

### Prerequisites

### Installation
