import os
import re
import sys
import csv
import copy
import json
import time
import math
import base64
import random
import logging
import argparse
import threading
import traceback
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from glob import glob
from tqdm import tqdm
from datetime import datetime
from logging.handlers import RotatingFileHandler

from utils.logger import setup_logger
from utils.metric import calculate_metrics
from utils.dataset import encode_image
from utils.dataset import load_sequences_from_json

# from utils.prompt_builder import prompt_builder

import google.generativeai as genai


def load_dataset_config(config_path):
    """
    Load dataset configuration from CSV file

    Returns:
        dict: trajectory_id -> intention mapping
    """
    intention_map = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                intention_map[row["trajectory_id"]] = row["intention"]
    except FileNotFoundError:
        print(f"Warning: Dataset config file not found at {config_path}")
        return {}
    return intention_map


def load_api_config(config_path):
    """
    Load API configuration from JSON file

    Returns:
        dict: API configuration
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: API config file not found at {config_path}")
        return {}


def get_intention_from_config(
    trajectory_id, intention_map, fallback_instruction="Complete the task"
):
    """
    Get intention from config map or use fallback

    Args:
        trajectory_id: ID to look up
        intention_map: Dict mapping IDs to intentions
        fallback_instruction: Default instruction if not found

    Returns:
        str: Task instruction
    """
    if trajectory_id in intention_map:
        return intention_map[trajectory_id]

    # Fallback: extract from trajectory_id if possible
    print(f"Warning: No intention found for {trajectory_id}, using fallback")
    return fallback_instruction


GENERAL_INSTRUCTION = """[General Instruction]
You are a friendly AI coach with balanced sensitivity to task focus and a neutral communication style. 
The user's current intention is provided as [intention: {task_name}]. 
Help users stay mindful of their task while providing feedback that matches your assigned tone and sensitivity. 
Consider the specific nature of their task when giving suggestions and feedback. 
For example, given a task of shopping, user may watching some reviews of several items. Or, given a task of writing a report, user may discuss with peers."""


CONTEXT_INSTRUCTION_WITHOUT_FORMAT = """[Key Instruction for Evaluating Relevance]
- **Examine Details**: Look for specific details in the provided information such as context of the conversation or title of the video on YouTube.
- **Analyze Context Beyond Keywords**: Do not judge content solely based on its surface-level category (e.g., chat, video, email). Instead, determine if it serves the task.
- **Bridge Indirect Relevance**: If an activity indirectly supports the task (e.g., searching, communicating, or researching), recognize its role instead of marking it as unrelated.
- **Be Certain for Scores**: Only label an activity as highly aligned or highly unrelated if there is clear evidence. If unsure, assign an intermediate value."""

CONTEXT_INSTRUCTION_WITH_FORMAT = """[Key Instruction for Evaluating Relevance]
- **Predict Intent**: Before judging anything, predict the intent of the user's behavior based on the provided information.
- **Examine Details**: Look for specific details in the provided information such as context of the conversation or title of the video on YouTube.
- **Analyze Context Beyond Keywords**: Do not judge content solely based on its surface-level category (e.g., chat, video, email). Instead, determine if it serves the task.
- **Bridge Indirect Relevance**: If an activity indirectly supports the task (e.g., searching, communicating, or researching), recognize its role instead of marking it as unrelated.
- **Be Certain for Scores**: Only label an activity as highly aligned or highly unrelated if there is clear evidence. If unsure, assign an intermediate value."""

SCORING_GUIDELINE_DISCRETE = """[Scoring Guidelines]
0: Perfect match with your task   
- Clearly aligned with your task   
- Exactly what you need   
- Core learning material
- Right content   
- Right context

1: Absolutely unrelated to your task (100% certain)   
- Gaming during study
- Social media during work
- Technically unrelated but might feel productive   
- Random browsing"""

SCORING_GUIDELINE_PROBABILITY = """[Scoring Guidelines]
0.0: Perfectly relevant 
- Clearly aligned with your task 
- Example: Writing a report, coding for a project, or shopping for a specific item on e-commerce

0.2: Mostly relevant  
- Indirectly relevant but necessary (e.g., searching, communication, reference gathering)
- Example: Watching a video tutorial on the same topic, reading a related article, or discussing with peers about the task

0.4: Somewhat relevant  
- Indirectly helpful but not essential 
- Example: Watching a review video or engaging in a discussion that could be related but lacks clear context

0.6: Somewhat irrelevant
- Unclear if it supports or distracts 
- Example: Initial page of web browser, diverse thumbnails of YouTube videos, desktop video, finding a file of in Finder

0.8: Mostly irrelevant  
- Has little to do with the task but could still offer minimal benefits 
- Example: Watching a video having or casual discussions that slightly touch on the topic

1.0: Completely irrelevant  
- Clearly a distraction or off-topic activity  
- Example: Gaming during study, social media during work, random entertainment browsing"""


IMPORTANT_RULES = """[IMPORTANT Rules]
Return only the JSON object.  
"""

CLARIFICATION_CONTEXT = """[Clarification Context]
Additionally, given [intention: {task_name}] from the user, the below content provides possible activities that the user may perform, based on the clarification questions and answers.
Please use this context of augmented intention for more accurate classification.
{list_of_augmented_intention}"""

# Learning context from user feedback - Implicit intentions
LEARNING_FROM_FEEDBACK_CONTEXT = """[Reflection Context]
Furthermore, given [intention: {task_name}] from the user, the following context have been learned from the user's past feedback. 
Each reflected context is composed as: {{"implicit intentions learned from reflection" ("relevant description of user activity")}}. 
Please use this augmented intention context for more accurate classification.
{list_of_learned_intentions}"""

# Learning context from user feedback - Scoring rules
ADJUST_FROM_FEEDBACK_CONTEXT = """[Reflection Rules]
The following rules have been learned based on the user's past feedback. 
Each reflected rules is composed as: {{"scoring rule learned from reflection" ("relevant description of user activity")}}. 
{list_of_learned_rules}"""


def prompt_builder(args) -> str:
    prompt_text = ""

    prompt_text += GENERAL_INSTRUCTION + "\n\n"

    if args.clarify_intentions:
        prompt_text += CLARIFICATION_CONTEXT + "\n\n"

    if args.feedback_from_user:
        prompt_text += LEARNING_FROM_FEEDBACK_CONTEXT + "\n\n"
        prompt_text += ADJUST_FROM_FEEDBACK_CONTEXT + "\n\n"

    if args.context_consideration:
        if args.formatted_prediction:
            prompt_text += CONTEXT_INSTRUCTION_WITH_FORMAT + "\n\n"
        else:
            prompt_text += CONTEXT_INSTRUCTION_WITHOUT_FORMAT + "\n\n"

    if args.probabilistic_score:
        prompt_text += SCORING_GUIDELINE_PROBABILITY + "\n\n"
    else:
        prompt_text += SCORING_GUIDELINE_DISCRETE + "\n\n"

    prompt_text += "[Output Format]\n"
    prompt_text += "\{\n"
    if args.formatted_prediction:
        prompt_text += '"intent prediction": "",  // Predict the intent of the user using the specific format: [Verb] + [Object] + (Optional) [Context] (e.g., "Write an email to Amy for Tuesday meeting" or "Watch tutorial on YouTube).\n'
    prompt_text += '"reason": "",  // One clear sentence explicitly mentioning its relevance or irrelevance to the task.\n'
    if args.probabilistic_score:
        prompt_text += '"output": 0.0  // Score in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, where 0.0 is fully aligned and 1.0 is completely unrelated.\n'
    else:
        prompt_text += (
            '"output": 1,  // 0-1 scoring level 0 (aligned), 1 (distracted)\n'
        )
    prompt_text += "\}\n\n"

    prompt_text += IMPORTANT_RULES

    return prompt_text


def reflect(
    args,
    model,
    logger,
    stated_intention,
    image_path,
    previous_response,
):
    reflection_prompt_format = f"""You are a helpful assistant designed to reflect on your predictions with user's feedback.
Your goal is to output an implicit intention of the user, which has not been stated but should have been captured, to explain the user's activity in a way that aligns with the user's current task.
You need to analyze a situation where the user's feedback toward your previous judgment. 
For example, given a stated intention "Write a research report" and a screen description "Chatting with a colleague on Slack", you may have classified it as a distraction with the rationale "This appears to be casual conversation, not task-related."
Then, the user might provide 'dislike' to your judgement.
Therefore, you should reflect and output an implicit intention of the user, such as "Discuss with a colleague for sources for the report", which explains the user's activity in a way that aligns with the user's current task.

[Stated Intention]
{stated_intention}

[Your Response]
Low score of output indicates that you judged that the user's activity aligns with the user's intention.
Output Score: {previous_response.get("output", 0.0)}
Reason: {previous_response.get("reason", "")}

[User Feedback]
dislike (expressed when prediction=1 and actual_label=0, indicating false positive)

Now, reflect on why the user might have expressed such feedback. 
Think about what **implicit intention** or subtle task-related reasoning the user might have had, which you did not consider. 
Then, build a policy adjustment strategy to better align your future judgments with the user's task.
Especially, the policy adjustment should follow the format of "Output high/low alignment (low/high score output) for [specific activity with detailed contents] when detected"

Respond in **JSON format** with four keys:
- "analysis_assistant_response": judge whether your previous response was high alignment (low output score) or low alignment (high output score) with the user's intention
- "user_activity_description": a short sentence describing the activity shown in the screen image in noun form (e.g., "YouTube homepage with diverse video thumbnails", within 20 words)
- "analysis_user_feedback": two short sentences (within 10 words each) explaining what/why the user disliked your judgement of alignment (e.g., "User expressed dislike. The reason is because judgment was too strict")
- "user_implicit_intention_prediction": a short sentence (within 10 words) predicting an implicit intention of the user that aligns with the user's current activity, starting with a verb (e.g., "Watch review before purchase")

Only return the JSON object. Do not include any explanation or prefix text.
"""
    encoded_img = encode_image(image_path, console_logger)

    reflection_prompt = reflection_prompt_format
    reflection_prompt = reflection_prompt.replace(
        "{stated_intention}", stated_intention
    )

    logger.info(f"Reflection prompt: {reflection_prompt}")

    try:
        # Handle both APILoadBalancer and single model
        if hasattr(model, "models"):  # APILoadBalancer
            response = model.generate_content(reflection_prompt, encoded_img)
        else:  # Single model
            response = model.generate_content(
                [reflection_prompt, encoded_img], generation_config=generation_config
            )
        response_text = response.text.strip()
        logger.info(f"Raw Reflection Response:\n{response_text}")

        cleaned = process_response(response_text, mode="reflection")
        reflection_data = json.loads(cleaned)

        return reflection_data

    except:
        error_txt = traceback.print_exc()
        logger.error(f"Reflection failed: {error_txt}")

    return None


def simulate_user(
    args,
    model,
    logger,
    consecutive_distraction_count,
    task_instruction,
    image_path,
    label,
    response=None,
):
    reflection = None

    prediction = response["predicted_score"]

    # dislike for every false notification (False Positive only)
    if prediction == 1 and label == 0:
        reflection = reflect(
            args, model, logger, task_instruction, image_path, response
        )

    return consecutive_distraction_count, reflection


def process_response(json_str, mode="response"):
    """
    Clean and sanitize JSON string from LLM output

    Args:
        json_str: The raw JSON string to clean

    Returns:
        Cleaned JSON string that can be parsed
    """
    if not json_str:
        return "{}"

    cleaned = json_str.replace("```json", "").replace("```", "").strip()
    cleaned = cleaned.replace("'", '"').replace("'", '"').replace("'", '"')
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")
    if not cleaned.strip().startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.strip().endswith("}"):
        cleaned = cleaned + "}"
    cleaned = re.sub(r"(?m)^(\s*)(\w+)(\s*):(\s*)", r'\1"\2"\3:\4', cleaned)

    match = re.search(r"{.*?}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        json.loads(cleaned)
    except json.JSONDecodeError:
        if mode == "response":
            score_match = re.search(r'"?output"?\s*:\s*([0-9.]+)', cleaned)
            reason_match = re.search(
                r'"?reason"?\s*:\s*"([^"]*)"', cleaned
            ) or re.search(r'"?reason"?\s*:\s*\'([^\']*)\'', cleaned)

            predicted_score = float(score_match.group(1)) if score_match else 0.0
            reason = (
                reason_match.group(1) if reason_match else "Failed to extract reason"
            )

            cleaned = f'{{"output": {predicted_score}, "reason": "{reason}"}}'
        elif mode == "reflection":
            # Updated for 4-key reflection format
            analysis_response_match = re.search(
                r'"?analysis_assistant_response"?\s*:\s*"([^"]+)"', cleaned
            )
            activity_match = re.search(
                r'"?user_activity_description"?\s*:\s*"([^"]+)"', cleaned
            )
            feedback_match = re.search(
                r'"?analysis_user_feedback"?\s*:\s*"([^"]+)"', cleaned
            )
            intention_match = re.search(
                r'"?user_implicit_intention_prediction"?\s*:\s*"([^"]+)"', cleaned
            )

            analysis_response = (
                analysis_response_match.group(1)
                if analysis_response_match
                else "low alignment"
            )
            activity_description = (
                activity_match.group(1) if activity_match else "Unknown activity"
            )
            feedback_analysis = (
                feedback_match.group(1)
                if feedback_match
                else "User expressed dislike. Judgment was too strict"
            )
            intention_prediction = (
                intention_match.group(1)
                if intention_match
                else "Perform task-related activity"
            )

            cleaned = (
                f'{{"analysis_assistant_response": "{analysis_response}", '
                f'"user_activity_description": "{activity_description}", '
                f'"analysis_user_feedback": "{feedback_analysis}", '
                f'"user_implicit_intention_prediction": "{intention_prediction}"}}'
            )

    return cleaned


def process_step(
    args,
    model,
    logger,
    image_path,
    task_instruction,
    prompt_text,
    clarification_intentions=None,
    learned_intentions=None,
    learned_rules=None,
):
    """
    Process a single image with the LLM

    Args:
        image_path: Path to the image to process
        task_instruction: The task instruction to use
        prompt_text: Prompt text to use for the model
        logger: Logger instance for logging
    """
    try:
        encoded_img = encode_image(image_path, console_logger)
        logger.info(f"Processing image: {os.path.basename(image_path)}")

        # Format the prompt by replacing the task placeholder
        formatted_prompt = prompt_text.replace("{task_name}", task_instruction)

        if (
            args.clarify_intentions
            and args.clarify_intentions != "None"
            and clarification_intentions
        ):
            formatted_prompt = formatted_prompt.replace(
                "{list_of_augmented_intention}",
                "\n".join([f"- {intent}" for intent in clarification_intentions]),
            )

        if args.feedback_from_user:
            # Stage 1: Apply learned intentions
            if learned_intentions and len(learned_intentions) > 0:
                intentions_text = "\n".join(
                    [f"- {intent}" for intent in learned_intentions]
                )
            else:
                intentions_text = "(No learned intentions yet)"
            formatted_prompt = formatted_prompt.replace(
                "{list_of_learned_intentions}",
                intentions_text,
            )

            # Stage 2: Apply learned rules
            if learned_rules and len(learned_rules) > 0:
                rules_text = "\n".join([f"- {rule}" for rule in learned_rules])
            else:
                rules_text = "(No learned rules yet)"
            formatted_prompt = formatted_prompt.replace(
                "{list_of_learned_rules}",
                rules_text,
            )

        logger.info(f"Formatted prompt: {formatted_prompt}")

        # Get response from model (supports both single model and load balancer)
        response = model.generate_content(
            [formatted_prompt, encoded_img], generation_config=generation_config
        )
        # elif "claude" in args.model_name:
        #     response = model.messages.create(
        #         max_tokens=1024,
        #         messages=[
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {
        #                         "type": "image",
        #                         "source": {
        #                             "type": "base64",
        #                             "media_type": encoded_img["mime_type"],
        #                             "data": encoded_img["data"],
        #                         },
        #                     },
        #                     {"type": "text", "text": formatted_prompt},
        #                 ],
        #             }
        #         ],
        #         model=args.model_name,
        #     )

        # Extract JSON from response - prevent binary output
        response_str = response.text if hasattr(response, "text") else "{}"
        logger.info(f"Raw response: {response_str}")

        try:
            json_str = process_response(response_str)
            result = json.loads(json_str)

            # log the result
            for key, value in result.items():
                logger.info(f"Result Key: {key}, Value: {value}")

            # postprocess
            if result["output"] < 0.0 or result["output"] > 1.0:
                result["output"] = max(0.0, min(1.0, result["output"]))
            valid_scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            result["output"] = min(
                valid_scores, key=lambda x: abs(x - result["output"])
            )

            # mark no error
            result["error"] = False

        except:
            result = {
                "output": 0.0,
                "reason": "",
                "error": True,
            }

        # Add detailed timing information
        # result["prompt_text"] = formatted_prompt  # Store prompt text
        result["raw_response"] = response_str  # Store raw response for debugging
        result["predicted_score"] = 1 if result["output"] >= 0.6 else 0

        return result
    except:
        error_log = traceback.format_exc()
        logger.error(f"Error processing image {image_path}: {error_log}")
        return {
            "output": 0.0,
            "predicted_score": 0,
            "reason": f"Error processing image {image_path}",
            "raw_response": "",
            "error": True,
        }


def process_trajectory(
    args,
    model,
    logger,
    task_instruction,
    image_paths,
    labels,
    task_ids,
    prompt_text="",
    clarification_intentions=None,
    results_dir="results",
    training_name="",
):
    """
    Process all tasks in a training file sequentially, with optional context

    Args:
        tasks: Dictionary of tasks to process
        task_instruction: The task instruction to use
        results_dir: Directory to save results
        use_context: Whether to use context between images
        training_name: Name of the training file being processed
        logger: Logger instance for logging
    """
    logger.info(f"Starting sequential processing of {len(image_paths)} images...")

    # Apply sampling if sample_rate > 1
    if args.sample_rate > 1:
        # Sample every N-th image
        sampled_indices = list(range(0, len(image_paths), args.sample_rate))
        original_count = len(image_paths)
        image_paths = [image_paths[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]
        task_ids = [task_ids[i] for i in sampled_indices]
        logger.info(
            f"📊 Sampling enabled: Processing {len(image_paths)}/{original_count} images (every {args.sample_rate}-th image)"
        )
    else:
        logger.info(f"📊 Processing all {len(image_paths)} images")

    # for feedback - 2-stage learning system
    learned_intentions = []  # Implicit intentions learned from reflection
    learned_rules = []  # Scoring rules learned from reflection
    consecutive_distraction_count = 0

    # Main loop
    results = []
    total_images = len(image_paths)
    processed_images = 0

    csv_file = os.path.join(results_dir, f"{training_name}_results.csv")
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "task_id",
            "image_path",
            "label",
            "output_score",
            "binary_score",
            "reason",
            "processing_time",
            # "prompt_text",
            "raw_response",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file is new
        if not csv_exists:
            writer.writeheader()
            logger.info(f"Created new results CSV file: {csv_file}")
        else:
            logger.info(f"Appending to existing results CSV file: {csv_file}")

        for i, (image_path, label, task_id) in enumerate(
            zip(image_paths, labels, task_ids)
        ):
            processed_images += 1

            # Process image with context from previous results (or None)
            result = process_step(
                args,
                model,
                logger,
                image_path,
                task_instruction,
                prompt_text=prompt_text,
                clarification_intentions=clarification_intentions,
                learned_intentions=learned_intentions,
                learned_rules=learned_rules,
            )

            # Store result
            result["task_id"] = task_id
            result["image_path"] = image_path
            result["label"] = label
            results.append(result)

            # Log progress to file (every 10% or when complete)
            if processed_images % max(1, total_images // 10) == 0:
                logger.info(f"Progress: {processed_images}/{total_images} images")

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "image_path": image_path,
                    "label": label,
                    "output_score": result["output"],
                    "binary_score": result["predicted_score"],
                    "reason": result["reason"],
                    # "prompt_text": result["prompt_text"],
                    "raw_response": result["raw_response"],
                }
            )
            csvfile.flush()

            # simulate feedback from the user
            if args.feedback_from_user:
                simulation_result = simulate_user(
                    args,
                    model,
                    logger,
                    consecutive_distraction_count,
                    task_instruction,
                    image_path,
                    label,
                    response=result,
                )
                consecutive_distraction_count, reflection = simulation_result
                if reflection:
                    # Extract 4-dimensional reflection data for 2-stage learning
                    activity_desc = reflection.get("user_activity_description", "")
                    implicit_intention = reflection.get(
                        "user_implicit_intention_prediction", ""
                    )
                    feedback_analysis = reflection.get("analysis_user_feedback", "")

                    # Stage 1: Learn implicit intentions
                    if implicit_intention and activity_desc:
                        intention_entry = f'"{implicit_intention}" ("{activity_desc}")'
                        learned_intentions.append(intention_entry)
                        logger.info(f"Learned intention: {intention_entry}")

                    # Stage 2: Learn scoring rules based on feedback analysis
                    if feedback_analysis and activity_desc:
                        # Generate scoring rule from feedback analysis
                        scoring_rule = f'"Output low score for {activity_desc} when task-related context detected" ("{feedback_analysis}")'
                        learned_rules.append(scoring_rule)
                        logger.info(f"Learned rule: {scoring_rule}")

    # save results
    combined_file = os.path.join(results_dir, "results.json")
    try:
        with open(combined_file, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save combined JSON results: {str(e)}")

    logger.info(f"All {total_images} images processed successfully.")
    logger.info(f"Results saved to CSV: {csv_file}")

    return results


def inference_single(
    args, model, json_file, main_results_dir, prompt_text, intention_map, run_info=None
):
    """
    Process a dataset point (a JSON file) by handling its tasks sequentially

    Args:
        json_file: Path to the JSON file to process
        main_results_dir: Main results directory
        prompt_text: Prompt format
        intention_map: Dict mapping trajectory IDs to intentions
        run_info: Dictionary with timestamp and logs_dir for this run
    """
    process_name = json_file.split("/")[-1].split(".")[0]

    # Create results directory
    results_dir = os.path.join(main_results_dir, process_name)
    os.makedirs(results_dir, exist_ok=True)

    # Logger
    if run_info and "logs_dir" in run_info:
        logs_dir = run_info["logs_dir"]
    else:
        logs_dir = f"{WORK_PATH}/logs"
        os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{process_name}.log")
    logger = setup_logger(process_name, log_file)

    # Load the data
    task_info, label_info = load_sequences_from_json(json_file)
    traj_0_id, traj_1_id, original_task_instruction, image_paths = task_info
    labels, task_ids = label_info
    logger.info(f"Total {len(labels)} images")

    # Get task instruction from config instead of folder name
    task_instruction = get_intention_from_config(
        traj_0_id, intention_map, original_task_instruction
    )
    logger.info(f"Using task instruction: {task_instruction}")
    logger.info(f"Original instruction from data: {original_task_instruction}")

    # clarification intentions
    if args.clarify_intentions and args.clarify_intentions != "None":
        with open(args.clarify_intentions, "r") as f:
            clarification_intentions = json.load(f)[traj_0_id]["augmented_intentions"]
    else:
        clarification_intentions = None

    # Main loop
    start_time = time.time()

    results = process_trajectory(
        args,
        model,
        logger,
        task_instruction,
        image_paths,
        labels,
        task_ids,
        results_dir=results_dir,
        prompt_text=prompt_text,
        clarification_intentions=clarification_intentions,
        training_name=process_name,
    )

    end_time = time.time()
    total_time = end_time - start_time

    # Log metrics
    metrics = calculate_metrics(results)
    metrics_file = os.path.join(results_dir, "metrics.json")
    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {str(e)}")

    logger.info(f"{'='*50}")
    logger.info(f"COMPLETED: {process_name}")
    logger.info(f" - Images: {metrics['total_images']:d}")
    logger.info(f" - Accuracy: {metrics['accuracy']:.2f}")
    logger.info(f" - Precision: {metrics['precision']:.2f}")
    logger.info(f" - Recall: {metrics['recall']:.2f}")
    logger.info(f" - F1 Score: {metrics['f1']:.2f}")
    logger.info(f" - Total: {total_time:.2f}s")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"{'='*50}")

    return results, metrics


def inference_parallel(
    args, model, json_files, intention_map, max_workers=30, prompt_text=""
):
    """
    Process multiple training files in parallel, but each file's tasks are processed sequentially

    Args:
        json_files: List of JSON files to process
        intention_map: Dict mapping trajectory IDs to intentions
        max_workers: Maximum number of workers (files to process in parallel)
        use_context: Whether to use context between images
    """
    results_dict = {}
    all_metrics = {}

    # Set experiment name or use timestamp
    if args.experiment_name:
        experiment_id = args.experiment_name
    else:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory
    main_results_dir = f"{WORK_PATH}/results/{experiment_id}"
    os.makedirs(main_results_dir, exist_ok=True)
    logs_base_dir = f"{WORK_PATH}/logs"
    os.makedirs(logs_base_dir, exist_ok=True)
    logs_dir = f"{logs_base_dir}/{experiment_id}"
    os.makedirs(logs_dir, exist_ok=True)

    # Main progress log file
    main_log_file = os.path.join(logs_dir, "main.log")
    main_logger = setup_logger(f"main_{experiment_id}", main_log_file)
    main_logger.info("Arguments:")
    for key, value in vars(args).items():
        main_logger.info(f"  {key}: {value}")

    # Log sampling information
    if args.sample_rate > 1:
        main_logger.info(
            f"📊 SAMPLING ENABLED: Processing every {args.sample_rate}-th image to reduce costs"
        )
    else:
        main_logger.info(f"📊 Processing all images (no sampling)")

    # configure the number of workers based on the dataset size and API capacity
    total_dataset = len(json_files)
    actual_workers = min(max_workers, total_dataset)

    # Log run information
    run_info = {"timestamp": experiment_id, "logs_dir": logs_dir}
    main_logger.info(f"STARTING VALIDATION WITH {total_dataset} TRAINING FILES")
    main_logger.info(f"Max parallel files: {max_workers}")
    main_logger.info(f"Results directory: {main_results_dir}")
    main_logger.info(f"Logs directory: {logs_dir}")
    main_logger.info(f"Using {actual_workers} workers for parallel processing")

    # main run
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_file = {}
        for json_file in json_files:
            future = executor.submit(
                inference_single,
                args,
                model,
                json_file,
                main_results_dir,
                prompt_text,
                intention_map,
                run_info,
            )
            future_to_file[future] = json_file

            # Brief pause to prevent log output overlap
            time.sleep(0.1)

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            json_file = future_to_file[future]
            process_name = json_file.split("/")[-1].split(".")[0]

            try:
                results, metrics = future.result()
                results_dict[process_name] = results
                all_metrics[process_name] = metrics

                # Log to main log
                main_logger.info(f"Completed {process_name}")
                main_logger.info(f" - Images: {metrics['total_images']}")
                main_logger.info(f" - Accuracy: {metrics['accuracy']:.2f}")
                main_logger.info(f" - Precision: {metrics['precision']:.2f}")
                main_logger.info(f" - Recall: {metrics['recall']:.2f}")

            except:
                error_log = traceback.format_exc()
                main_logger.error(f"Error processing file {process_name}: {error_log}")

    # Save results
    with open(f"{main_results_dir}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    summary_csv = os.path.join(main_results_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "training_file",
                "total_images",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        )

        for process_name, metrics in all_metrics.items():
            writer.writerow(
                [
                    process_name,
                    metrics.get("total_images", "N/A"),
                    metrics.get("true_positive", "N/A"),
                    metrics.get("true_negative", "N/A"),
                    metrics.get("false_positive", "N/A"),
                    metrics.get("false_negative", "N/A"),
                    metrics.get("accuracy", "N/A"),
                    metrics.get("precision", "N/A"),
                    metrics.get("recall", "N/A"),
                    metrics.get("f1", "N/A"),
                ]
            )

    main_logger.info(f"Summary saved to: {summary_csv}")

    return


def parse_arg():
    """
    Parse command-line arguments for evaluation options.
    """
    parser = argparse.ArgumentParser(description="Evaluation Options")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Name of the model to use (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--probabilistic_score",
        type=str2bool,
        default=True,
        help="Whether to use probabilistic scoring (default: True)",
    )
    parser.add_argument(
        "--context_consideration",
        type=str2bool,
        default=True,
        help="Whether to consider context during processing (default: True)",
    )
    parser.add_argument(
        "--formatted_prediction",
        type=str2bool,
        default=False,
        help="Whether to format the prediction outputs (default: False)",
    )
    parser.add_argument(
        "--clarify_intentions",
        type=str,
        default=None,  # f"{WORK_PATH}/experiments/asset/augment_stated_intentions.json",
        help="Specify the path to the file containing augmented stated intentions for clarification questions (default: None)",
    )
    parser.add_argument(
        "--feedback_from_user",
        type=str,
        default=False,
        help="Whether to test the feedback from the user or not",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help=(
            "Path to dataset configuration CSV file "
            "(default: intention_bench/dataset/annotations/metadata/config/intentions.csv)"
        ),
    )
    parser.add_argument(
        "--api_config",
        type=str,
        default=None,
        help="Path to API configuration JSON file (default: None, uses config/api_config.json)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to synthetic data directory "
            "(default: intention_bench/dataset/annotations/mixed_sessions/raw_jsons)"
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for experiment results directory and logs (default: None, uses timestamp)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="Process every N-th image (default: 1, process all images. 2 = every other image, 3 = every third image, etc.)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()

    # Main logger setup - for console output
    console_logger = logging.getLogger("console")
    console_logger.setLevel(logging.INFO)
    if not console_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_logger.addHandler(console_handler)

    # Set required environment variables
    if "INTENTIONAL_COMPUTING_HOME" not in os.environ:
        os.environ["INTENTIONAL_COMPUTING_HOME"] = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        console_logger.info(
            f"Environment variable INTENTIONAL_COMPUTING_HOME is not set, so it is set to the default value: {os.environ['INTENTIONAL_COMPUTING_HOME']}"
        )

    WORK_PATH = os.environ["INTENTIONAL_COMPUTING_HOME"]

    # Load API configuration
    api_config_path = (
        args.api_config or f"{WORK_PATH}/experiments/config/api_config.json"
    )
    api_config = load_api_config(api_config_path)

    gemini_config = api_config.get("gemini", {}) if api_config else {}
    api_key = gemini_config.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    project_id = gemini_config.get("project_id") or os.environ.get("GCP_PROJECT_ID", "")
    location = gemini_config.get("location", "us-east5")

    if not api_key or not project_id:
        raise EnvironmentError(
            "Missing Gemini API credentials. Provide api_key and project_id via "
            "api_config.json or environment variables (GEMINI_API_KEY, GCP_PROJECT_ID)."
        )

    # Persist credentials in environment for downstream helpers
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GCP_PROJECT_ID"] = project_id
    os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
    os.environ.setdefault("GEMINI_LOCATION", location)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model_name)

    gen_config = api_config.get(
        "generation_config",
        {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 256,
        },
    )
    generation_config = genai.GenerationConfig(**gen_config)

    # Simple prompt test to verify API connection
    response = model.generate_content("What is 2+2?", generation_config=generation_config)
    console_logger.info("API Test Response: " + response.text)

    # Load dataset configuration
    dataset_config_path = (
        args.dataset_config
        or f"{WORK_PATH}/intention_bench/dataset/annotations/metadata/config/intentions.csv"
    )
    intention_map = load_dataset_config(dataset_config_path)
    console_logger.info(f"Loaded {len(intention_map)} intention mappings from config")

    # Get all training files from specified directory
    data_dir = (
        args.data_dir
        or f"{WORK_PATH}/intention_bench/dataset/annotations/mixed_sessions/raw_jsons"
    )

    # Support different naming patterns
    training_files = []
    patterns = [
        f"{data_dir}/training_*.json",  # Old format
        f"{data_dir}/type0_*.json",  # New format type 0
        f"{data_dir}/type1_*.json",  # New format type 1
        f"{data_dir}/type2_*.json",  # New format type 2
    ]

    for pattern in patterns:
        training_files.extend(sorted(glob(pattern)))

    # Remove duplicates and sort
    training_dataset = sorted(list(set(training_files)))
    total_dataset = len(training_dataset)
    console_logger.info(f"Found {total_dataset} training files in {data_dir}")

    # Prompt setting
    prompt_text = prompt_builder(args)

    # Set experiment name for results and logs
    if args.experiment_name:
        console_logger.info(f"🏷️ Experiment name: {args.experiment_name}")
    else:
        console_logger.info(f"🕐 Using timestamp for experiment name")

    # Process all training files in parallel (each file's tasks are processed sequentially)
    inference_parallel(
        args,
        model,
        training_dataset,
        intention_map,
        max_workers=args.max_workers,  # Use command line argument for worker count
        prompt_text=prompt_text,
    )

    console_logger.info("All task validation complete!")
