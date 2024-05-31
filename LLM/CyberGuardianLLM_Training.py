import argparse
from CyberGuardinaLLM_args import parse_args
from CyberGuardianLLM import CyberGuardianLLM
import Data.dataSettings as dataSettings
import pathlib
import os

def main():
    args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_TRAINING"]))
    cg = CyberGuardianLLM(args)
    cg.do_training()

if __name__ == "__main__":
    main()

