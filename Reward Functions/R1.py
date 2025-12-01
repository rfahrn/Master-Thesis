import numpy as np
import re
from typing import List, Dict, Any, Optional
IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Lower start for medical safety
MAP_WEIGHT = 0.8          # Main mAP reward
LOCALIZATION_WEIGHT = 0.1  # Bonus  high-quality boxes
COVERAGE_WEIGHT = 0.1      # Bonus  finding all lesions
NO_FINDING_REWARD = 0.3   # "no findings" 
EMPTY_PENALTY = -0.1       # penalty for missing all boxes
VALIDATION_METRICS = []
def extract_bounding_boxes(text: str) -> List[List[float]]:
    """Extract normalized bounding boxes from text."""
    if '<answer>' in text.lower():
        match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()

    #check "no findings"
    no_finding_phrases = [
        'no finding', 'no abnormal', 'normal', 'clear',
        'no lesion', 'unremarkable', 'no acute'
    ]

    text_lower = text.lower()
    if any(phrase in text_lower for phrase in no_finding_phrases):
        if not re.search(r'\[\s*[\d\.]+\s*,', text):
            return []

    # Extract boxes
    pattern = r'\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]'
    boxes = []
    for match in re.findall(pattern, text):
        try:
            box = [float(x) for x in match]
            if all(0 <= c <= 1 for c in box) and box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        except:
            continue

    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision (AP) - standard implementation."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

#iou-ranked ap computation
def compute_ap_at_iou_ranked(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float
) -> float:
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    # Special cases
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0

    # Compute IoU matrix
    ious = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            ious[i, j] = compute_iou(pred, gt)

    # Sort predictions by best IoU (using IoU as pseudo-confidence)
    best_ious = np.max(ious, axis=1)
    sorted_indices = np.argsort(-best_ious)  # Descending order

    # Process predictions in IoU-ranked order
    matched_gt = set()
    true_positives = np.zeros(n_pred)

    for rank_idx, pred_idx in enumerate(sorted_indices):
        best_gt_idx = np.argmax(ious[pred_idx, :])
        best_iou = ious[pred_idx, best_gt_idx]

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            true_positives[rank_idx] = 1  # Use rank index, not pred index
            matched_gt.add(best_gt_idx)

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(1 - true_positives)

    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # Compute proper AP
    ap = compute_average_precision(recall, precision)

    # Add soft bonus for continuous gradients (small contribution)
    # This rewards higher IoUs even above threshold
    if len(matched_gt) > 0:
        matched_ious = []
        for pred_idx in sorted_indices[:len(matched_gt)]:
            best_gt_idx = np.argmax(ious[pred_idx, :])
            if ious[pred_idx, best_gt_idx] >= iou_threshold:
                matched_ious.append(ious[pred_idx, best_gt_idx])

        if matched_ious:
            # Small bonus for IoU quality (0-10% boost based on how much above threshold)
            quality_factor = np.mean([(iou - iou_threshold) / (1 - iou_threshold)
                                     for iou in matched_ious if iou > iou_threshold])
            ap = min(1.0, ap * (1 + 0.1 * quality_factor))

    return ap


def compute_matches_at_iou_threshold(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float
) -> Dict[str, Any]:
    """
    Match predictions to ground truth at a specific IoU threshold.
    Uses IoU-ranked greedy matching.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 or n_gt == 0:
        return {
            'tp': 0,
            'fp': n_pred,
            'fn': n_gt,
            'tp_ious': [],
            'mean_tp_iou': 0.0
        }

    # Compute IoU matrix
    ious = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            ious[i, j] = compute_iou(pred, gt)

    # Sort predictions by best IoU (descending)
    best_ious = np.max(ious, axis=1)
    sorted_preds = np.argsort(-best_ious)

    # Greedy matching in IoU-ranked order
    matched_gt = set()
    tp_count = 0
    tp_ious = []

    for pred_idx in sorted_preds:
        available_gt = [j for j in range(n_gt) if j not in matched_gt]

        if not available_gt:
            break

        best_gt = max(available_gt, key=lambda j: ious[pred_idx, j])
        best_iou = ious[pred_idx, best_gt]

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            tp_count += 1
            tp_ious.append(best_iou)

    return {
        'tp': tp_count,
        'fp': n_pred - tp_count,
        'fn': n_gt - tp_count,
        'tp_ious': tp_ious,
        'mean_tp_iou': np.mean(tp_ious) if tp_ious else 0.0
    }

def compute_map_reward(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Compute IoU-ranked mAP reward.
    Uses proper AP calculation with IoU-based ranking.
    """
    if iou_thresholds is None:
        iou_thresholds = IOU_THRESHOLDS

    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    # Special case: True negative (both empty)
    if n_pred == 0 and n_gt == 0:
        return {
            'reward': NO_FINDING_REWARD,
            'map': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'mean_iou': 1.0,
            'case': 'true_negative'
        }

    # Special case: Complete miss (no predictions when GT exists)
    if n_pred == 0 and n_gt > 0:
        return {
            'reward': EMPTY_PENALTY,
            'map': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'mean_iou': 0.0,
            'case': 'complete_miss'
        }

    # Special case: False positives (predictions when no GT)
    if n_pred > 0 and n_gt == 0:
        # Penalize based on number of false positives
        penalty = EMPTY_PENALTY * (1 + 0.1 * n_pred)  # More FPs = worse
        return {
            'reward': max(-0.3, penalty),  # Cap maximum penalty
            'map': 0.0,
            'precision': 0.0,
            'recall': 1.0,  # No GT to miss
            'mean_iou': 0.0,
            'case': 'false_positives'
        }

    # Compute IoU-ranked AP at each threshold
    ap_scores = []
    all_tp_ious = []

    for iou_thresh in iou_thresholds:
        ap = compute_ap_at_iou_ranked(pred_boxes, gt_boxes, iou_thresh)
        ap_scores.append(ap)

        # Collect IoUs for quality bonus
        match_result = compute_matches_at_iou_threshold(pred_boxes, gt_boxes, iou_thresh)
        all_tp_ious.extend(match_result['tp_ious'])

    # Mean Average Precision (main reward component)
    map_score = np.mean(ap_scores)

    # Localization quality bonus (rewards high IoU)
    mean_iou = np.mean(all_tp_ious) if all_tp_ious else 0.0
    # Smooth bonus using sigmoid
    localization_bonus = (1 / (1 + np.exp(-10 * (mean_iou - 0.6)))) * LOCALIZATION_WEIGHT

    # Coverage bonus (rewards finding most/all lesions)
    strict_match = compute_matches_at_iou_threshold(pred_boxes, gt_boxes, iou_thresholds[-1])
    coverage = strict_match['tp'] / n_gt if n_gt > 0 else 0.0
    # Smooth coverage bonus
    coverage_bonus = (1 / (1 + np.exp(-10 * (coverage - 0.5)))) * COVERAGE_WEIGHT

    # Penalty for too many predictions (prevent box spamming)
    overprediction_ratio = n_pred / n_gt if n_gt > 0 else n_pred
    if overprediction_ratio > 2:  # More than 2x predictions vs GT
        overprediction_penalty = -0.05 * min(1, (overprediction_ratio - 2) / 3)
    else:
        overprediction_penalty = 0

    # Final reward composition
    base_reward = map_score * MAP_WEIGHT
    total_reward = base_reward + localization_bonus + coverage_bonus + overprediction_penalty

    # Get precision/recall at standard IoU=0.5 for reporting
    std_match = compute_matches_at_iou_threshold(pred_boxes, gt_boxes, 0.5)
    precision_at_50 = std_match['tp'] / n_pred if n_pred > 0 else 0.0
    recall_at_50 = std_match['tp'] / n_gt if n_gt > 0 else 0.0

    return {
        'reward': float(np.clip(total_reward, -0.3, 1.0)),
        'map': map_score,
        'precision': precision_at_50,
        'recall': recall_at_50,
        'mean_iou': mean_iou,
        'coverage': coverage,
        'ap_per_threshold': ap_scores,
        'overprediction_penalty': overprediction_penalty,
        'case': 'standard'
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Main entry point for VeRL RewardManager.

    Uses IoU-ranked AP for a principled reward without confidence scores.
    """
    # Extract boxes
    pred_boxes = extract_bounding_boxes(solution_str)
    gt_boxes = extract_bounding_boxes(ground_truth)

    # Compute IoU-ranked mAP reward
    result = compute_map_reward(pred_boxes, gt_boxes)
    reward = result['reward']

    # Store validation metrics if in validation mode
    if extra_info and (extra_info.get('split') == 'val' or extra_info.get('mode') == 'validation'):
        # For validation, use the same IoU-ranked AP at 0.5
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_pred == 0 and n_gt == 0:
            map_at_50 = 1.0
            avg_iou = 1.0
        elif n_pred == 0 or n_gt == 0:
            map_at_50 = 0.0
            avg_iou = 0.0
        else:
            # Compute IoU matrix
            ious = np.zeros((n_pred, n_gt))
            for i, pred in enumerate(pred_boxes):
                for j, gt in enumerate(gt_boxes):
                    ious[i, j] = compute_iou(pred, gt)

            # Average IoU across all pairs
            avg_iou = np.sum(ious) / (n_pred * n_gt)

            # Compute IoU-ranked AP at 0.5
            map_at_50 = compute_ap_at_iou_ranked(pred_boxes, gt_boxes, 0.5)

        val_metrics = {
            'mAP_0.5': map_at_50,
            'avg_iou': avg_iou,
            'precision@0.5': result['precision'],
            'recall@0.5': result['recall'],
            'difficulty': extra_info.get('difficulty_level', 'unknown'),
            'num_boxes': len(gt_boxes),
            'reward': reward
        }

        VALIDATION_METRICS.append(val_metrics)

        # Log periodically
        if len(VALIDATION_METRICS) % 100 == 0:
            recent = VALIDATION_METRICS[-100:]
            avg_map = np.mean([m['mAP_0.5'] for m in recent])
            avg_iou_val = np.mean([m['avg_iou'] for m in recent])
            print(f"[R1 IoU-ranked] Last 100: mAP@0.5={avg_map:.3f}, avg_iou={avg_iou_val:.3f}")

    return float(max(0.05, reward))