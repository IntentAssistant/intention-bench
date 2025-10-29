import os
import re
import json
import time
import math
import random
import base64
from glob import glob
from PIL import Image
from io import BytesIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # intention_bench/code/utils -> repo root
WORK_PATH = REPO_ROOT

TRAJECTORY_SEARCH_ROOTS = [
    WORK_PATH / "intention_bench" / "dataset" / "images",
    WORK_PATH / "intention_bench" / "dataset" / "focused_session_data",  # legacy
    WORK_PATH / "data" / "trajectory_data",  # legacy
]


def _find_trajectory_dirs(trajectory_code: str):
    matches = []
    for root in TRAJECTORY_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        pattern = str(root / f"*{trajectory_code}*")
        matches.extend(glob(pattern))
    return matches


def get_task_instruction(trajectory_code):
    directories = _find_trajectory_dirs(trajectory_code)
    if not directories:
        raise FileNotFoundError(
            f"Trajectory '{trajectory_code}' not found in known data directories."
        )
    folder_name = Path(directories[0]).name
    if "] " in folder_name:
        return folder_name.split("] ", 1)[1]
    return folder_name


def get_trajectory_dir(trajectory_code):
    directories = _find_trajectory_dirs(trajectory_code)
    return directories[0] if directories else None


def load_sequences_from_json(json_file):
    """
    Load image sequences from JSON files in the specified directory and organize them by task
    """
    with open(json_file, "r") as f:
        sequence_data = json.load(f)

    task_instruction = get_task_instruction(sequence_data["trajectory_0"])
    traj_0_id = sequence_data.get("trajectory_0")
    traj_1_id = sequence_data.get("trajectory_1", None)
    traj_0_dir = get_trajectory_dir(sequence_data["trajectory_0"])
    traj_1_dir = get_trajectory_dir(sequence_data["trajectory_1"])

    # Preprocess
    image_paths = []
    labels = []
    task_ids = []
    for file_name, label in zip(sequence_data["trajectories"], sequence_data["labels"]):
        if traj_1_dir:
            full_path = traj_0_dir if label == 0 else traj_1_dir
            full_path = os.path.join(full_path, file_name)
        else:
            full_path = os.path.join(traj_0_dir, file_name)

        image_paths.append(full_path)
        labels.append(label)

        if label == 0:
            task_ids.append(traj_0_id)
        else:
            task_ids.append(traj_1_id)

    return (traj_0_id, traj_1_id, task_instruction, image_paths), (labels, task_ids)


def encode_image(image_path, console_logger):
    """
    Encode an image file to base64 format for API requests

    Args:
        image_path: Path to the image file

    Returns:
        An object with mime_type and data that can be used in API requests
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Encode binary data to base64 but don't output it
        encoded_data = base64.b64encode(image_data).decode("utf-8")
        return {"mime_type": "image/jpeg", "data": encoded_data}
    except Exception as e:
        console_logger.error(f"Error encoding image {image_path}: {str(e)}")
        # Don't output binary data even in case of error
        return {"mime_type": "image/jpeg", "data": ""}
