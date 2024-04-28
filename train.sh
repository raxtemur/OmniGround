#!/bin/bash

TORCH_DISTRIBUTED_DEBUG=DETAIL python3 OmniFusion/train_src/train_sft.py --config OmniFusion/train_src/configs/config-sft-micro-gr.json