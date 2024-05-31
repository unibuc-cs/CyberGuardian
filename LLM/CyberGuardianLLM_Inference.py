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
    cg.do_inference()

    messages = [
        {"role": "system", "content": "You are an expert in cybersecurity"},
        {"role": "user", "content": "What is cybersecurity?"},
    ]

    cg.test_model(messages)


if __name__ == "__main__":
    main()