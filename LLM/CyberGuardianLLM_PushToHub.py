import gc
import os
import argparse
import json
import pathlib

from CyberGuardinaLLM_args import parse_args
from CyberGuardianLLM import CyberGuardianLLM
import Data.dataSettings as dataSettings

def main():
    # Load default arguments
    args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_INFERENCE"]))
    cg = CyberGuardianLLM(args)
    cg.do_inference(push_to_hub=True)

if __name__ == "__main__":
    main()