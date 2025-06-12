import datetime
import logging
import os
import torch
from openai import OpenAI
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
    )
import json
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

current_path = os.path.dirname(__file__)
main_path = os.path.abspath(os.path.join(current_path, os.pardir))


def mkdirs(dirpath):
    try: os.makedirs(dirpath)
    except Exception: pass


def set_seed(seed):
    """
    Random seed to ensure reproducibility
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def set_logger(args, combo_id):

    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M-%S')
    
    log_file_path = os.path.join(main_path, "results", f"{log_time}_{combo_id}.log")
    #log_path = f"{log_time}_{combo_id}.log"
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        filemode="w",
    )

    logger = logging.getLogger()
    return log_time, logger



