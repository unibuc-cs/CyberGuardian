import gc
import os
import argparse
import json
import pathlib

import Data.dataSettings as dataSettings
from projsecrets import project_path

from CyberGuardinaLLM_args import parse_args
from CyberGuardianLLM import CyberGuardianLLM

def main():
    # Load default arguments
    path_to_inference_params = pathlib.Path(os.environ["LLM_PARAMS_PATH_INFERENCE"])
    args = parse_args(with_json_args=path_to_inference_params )
    cg = CyberGuardianLLM(args)
    cg.do_inference(push_to_hub=False)

    messages = [
        {"role": "system", "content": "You are an expert in cybersecurity"},
        {"role": "user", "content": "What is cybersecurity?"},
    ]

    cg.test_model(messages)


if __name__ == "__main__":
    main()

