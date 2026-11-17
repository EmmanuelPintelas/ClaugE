"""Default locations of the training engine.

run_claugE.py and run_ablations.py always pass the pool folder (--out) and the dataset
(--data) to build_pool.py, so these defaults matter only when build_pool.py is run by hand:

    CLAUGE_POOLS        where pools are found (default: methods/ClaugE/pools)
    CLAUGE_DATA         packed datasets (default: <repository>/data)
    CLAUGE_BUILD_POOLS  where build_pool.py writes new pools (default: CLAUGE_POOLS)
"""
from __future__ import annotations

import os

ENGINE = os.path.dirname(os.path.abspath(__file__))
CLAUGE = os.path.dirname(ENGINE)
REPOSITORY = os.path.dirname(os.path.dirname(CLAUGE))

POOLS = os.environ.get("CLAUGE_POOLS", os.path.join(CLAUGE, "pools"))
DATA = os.environ.get("CLAUGE_DATA", os.path.join(REPOSITORY, "data"))
BUILD_POOLS = os.environ.get("CLAUGE_BUILD_POOLS", POOLS)
