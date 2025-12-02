"""
Grounding Data Preprocessing for VeRL
"""

import argparse
import os
import json
import re
import datasets
from pathlib import Path
from typing import List, Dict, Any

DATA_SOURCE = "medical_grounding"
# PROMPT_SUFFIX = " First output your analysis in <think> </think> tags and then output the final answer in <answer> </answer> tags. If you output bounding boxes make sure you normalize them [0,1]."
PROMPT_SUFFIX = "Provide your answer inside <answer>...</answer> tags. For bounding boxes: use format [x1, y1, x2, y2] normalized to [0,1] with x1<x2 and y1<y2."
MIN_PIXELS = 1024
MAX_PIXELS = 451584

DIFFICULTY_THRESHOLDS = {
    'easy': (0, 1),
    'medium': (2, 3),
    'hard': (4, float('inf'))
}

def extract_ground_truth(gpt_response: str) -> str:
    """Extract ground truth from GPT response."""
    match = re.search(r"<answer>(.*?)</answer>", gpt_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return gpt_response.strip()


def count_bounding_boxes(text: str) -> int:
    """Count number of bounding boxes in text."""
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    box_pattern = rf"\[\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\]"
    return len(re.findall(box_pattern, text))


def classify_difficulty(num_boxes: int) -> str:
    """Classify difficulty based on number of bounding boxes."""
    for difficulty, (min_boxes, max_boxes) in DIFFICULTY_THRESHOLDS.items():
        if min_boxes <= num_boxes <= max_boxes:
            return difficulty
    return 'hard'


def has_findings(ground_truth: str) -> bool:
    """Check if there are actual bounding boxes."""
    if not ground_truth or not ground_truth.strip():
        return False

    gt_lower = ground_truth.lower()
    no_finding_phrases = [
        "no finding", "no abnormality", "no abnormalities",
        "no detectable lesions", "no lesions", "not detected",
        "clean bill of health", "no visible abnormalities"
    ]
    if any(phrase in gt_lower for phrase in no_finding_phrases):
        return False

    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    box_pattern = rf"\[\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*,\s*{NUM}\s*\]"
    return bool(re.search(box_pattern, ground_truth))


def normalize_labels(labels) -> List[str]:
    if labels is None:
        return []

    if not labels:
        return []

    normalized = []
    for label in labels:
        if label is None:
            continue

        # Basic normalization
        norm = label.strip().lower()
        norm = norm.replace("_", " ").replace("-", " ")

        # Handle common variations
        synonym_map = {
            "no finding": "No finding",
            "no abnormality": "No finding",
            "no abnormalities": "No finding",
        }

        # Apply synonym mapping
        for key, value in synonym_map.items():
            if key in norm:
                norm = value
                break
        else:
            # Title case for consistency
            norm = ' '.join(word.capitalize() for word in norm.split())

        normalized.append(norm)

    return normalized


def make_map_fn(split: str, add_suffix: bool = True):
    """Convert LLaVA format to VeRL format (FAST version - no validation)."""
    def proc(example: Dict[str, Any], idx: int):
        try:
            # Get image path (make absolute)
            image_path = example.get('image', '')
            if not image_path:
                return None

            if not image_path.startswith('/'):
                image_path = os.path.abspath(image_path)

            img_entry = {
                "image": f"file://{image_path}",
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            }

            # Extract conversations
            conversations = example.get("conversations", [])
            if len(conversations) < 2:
                return None

            human_msg = conversations[0].get("value", "")

            # Add <image> token if missing
            if "<image>" not in human_msg:
                human_msg = f"<image>\n{human_msg}"

            # Add prompt suffix
            if add_suffix:
                human_msg = human_msg + PROMPT_SUFFIX

            prompt_msg = {
                "role": "user",
                "content": human_msg
            }

            # Extract ground truth
            gpt_response = conversations[1].get("value", "")
            ground_truth = extract_ground_truth(gpt_response)

            # Count boxes and classify difficulty
            num_boxes = count_bounding_boxes(ground_truth)
            difficulty = classify_difficulty(num_boxes)

            # Get labels directly from original data (handles None gracefully)
            raw_labels = example.get("labels")  # Your data already has this!
            normalized_labels = normalize_labels(raw_labels)

            # Determine ability
            ability = "radiology_grounding"
            human_lower = human_msg.lower()
            if "locate the following" in human_lower or "find and localize" in human_lower:
                ability = "phrase_grounding"
            elif normalized_labels and len(normalized_labels) > 0 and normalized_labels != ["No finding"]:
                ability = "medical_grounding_classified"

            return {
                "data_source": DATA_SOURCE,
                "prompt": [prompt_msg],
                "images": [img_entry],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "original_id": example.get("id", f"sample_{idx}"),
                    "labels": normalized_labels,  # From your original data
                    "split": split,
                    "index": idx,
                    "has_findings": has_findings(ground_truth),
                    "num_boxes": num_boxes,
                    "difficulty_level": difficulty,
                },
            }
        except Exception as e:
            # Silently skip problematic examples
            return None

    return proc



def main(args):

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.easy_threshold != 1 or args.hard_threshold != 4:
        DIFFICULTY_THRESHOLDS['easy'] = (0, args.easy_threshold)
        DIFFICULTY_THRESHOLDS['medium'] = (args.easy_threshold + 1, args.hard_threshold - 1)
        DIFFICULTY_THRESHOLDS['hard'] = (args.hard_threshold, float('inf'))

    split_files = {
        "train": ["train.json"],
        "val": ["val.json"],
    }

    if args.include_test:
        split_files["test"] = ["test.json"]

    overall_stats = {
        'total': 0,
        'difficulty': {'easy': 0, 'medium': 0, 'hard': 0}
    }

    for split, filenames in split_files.items():
        json_path = None
        for filename in filenames:
            candidate = input_dir / filename
            if candidate.exists():
                json_path = candidate
                break

        try:
            with open(json_path) as f:
                rows = json.load(f)
        except Exception as e:
            continue

        ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
        processed_ds = ds.map(
            make_map_fn(split, add_suffix=not args.no_suffix),
            with_indices=True,
            num_proc=args.num_proc,
            desc=f"Converting {split} to VeRL format"
        )

        output_path = output_dir / f"{split}.parquet"
        processed_ds.to_parquet(str(output_path))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FAST medical grounding preprocessing for VeRL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_dir",
        default="./llava_datasets",
        help="Directory containing JSON files"
    )
    parser.add_argument(
        "--output_dir",
        default="./verl_data_grounding",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--no_suffix",
        action="store_true",
        help="Don't add <think> and <answer> prompt suffix"
    )
    parser.add_argument(
        "--include_test",
        action="store_true",
        help="Also process test split if available"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--easy_threshold",
        type=int,
        default=1,
        help="Max boxes for 'easy' difficulty"
    )
    parser.add_argument(
        "--hard_threshold",
        type=int,
        default=4,
        help="Min boxes for 'hard' difficulty"
    )
    parser.add_argument(
        "--save_examples",
        action="store_true",
        default=True,
        help="Save example outputs to JSON"
    )

    args = parser.parse_args()
    main(args)
