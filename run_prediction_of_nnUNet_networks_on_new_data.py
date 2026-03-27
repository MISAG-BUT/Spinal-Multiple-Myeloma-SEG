# ==========================================================
# Imports & nnU-Net environment setup
# ==========================================================

import shutil
import argparse
from os.path import join
import sys, os
# Import config for all paths and environment setup
import config
from config import NNUNET_REPO_PATH, NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS_ENV
# Set up nnU-Net environment and sys.path
def setup_nnunet_env():
    """Set up sys.path and nnU-Net environment variables from config."""    
    if NNUNET_REPO_PATH not in sys.path:
        sys.path.append(NNUNET_REPO_PATH)
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS_ENV

setup_nnunet_env()

from utils import *