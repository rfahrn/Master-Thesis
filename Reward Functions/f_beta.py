import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional


MIN_IOU = 0.2          # minimum IoU to consider a match (lower = more forgiving)
BETA = 1.5        # F-beta parameter (>1 prioritizes recall for medical)
NO_FINDING_BONUS = 0.2 # Reward  "no findings"


def extract_bounding_boxes(text: str) -> List[List[float]]:
    """
    Extract normalized bounding boxes from text.
    Simple and robust extraction handling various formats.
    """
    # Handle <answer> tags
    if '<answer>' in text.lower():
        match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # Check for "no findings" - return empty list
    no_finding_phrases = [
        'no finding', 'no abnormal', 'normal', 'clear',
        'no lesion', 'unremarkable', 'no acute'
    ]
    
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in no_finding_phrases):
        # Only return empty if there are truly no coordinates
        if not re.search(r'\[\s*[\d\.]+\s*,', text):
            return []
    
    # Extract boxes - simple regex for [x1, y1, x2, y2]
    pattern = r'\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]'
    
    boxes = []
    for match in re.findall(pattern, text):
        try:
            box = [float(x) for x in match]
            # Basic validation
            if all(0 <= c <= 1 for c in box) and box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        except:
            continue
    
    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Standard IoU calculation."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


# quality score function

def iou_to_quality(iou: float, min_iou: float = MIN_IOU) -> float:
    """
    Convert IoU to quality score with smooth partial credit.
    
    This simple function provides:
    - 0 credit for IoU < min_iou (too poor)
    - Quadratic growth for IoU ∈ [min_iou, 0.5] (partial credit)
    - Linear growth for IoU ∈ [0.5, 1.0] (good matches)
    
    Why this works well:
    - Smooth gradients (no discontinuities)
    - Partial credit encourages learning
    - Simple to understand and tune
    """
    if iou < min_iou:
        return 0.0
    elif iou < 0.5:
        # Quadratic scaling from 0 to 0.5
        normalized = (iou - min_iou) / (0.5 - min_iou)
        return 0.5 * (normalized ** 2)
    else:
        # Linear scaling from 0.5 to 1.0
        return iou


#greedy matching algorithm

def greedy_match(pred_boxes: List[List[float]], 
                 gt_boxes: List[List[float]],
                 min_iou: float = MIN_IOU) -> Dict[str, Any]:
    """
    Simple greedy matching algorithm.
    
    Algorithm:
    1. For each prediction, find its best GT match
    2. Match if IoU >= min_iou
    3. Each GT can only be matched once
    4. Return quality scores for matched boxes
    
    Why greedy works well enough:
    - Intuitive and easy to debug
    - Fast O(n*m) complexity
    - Good enough for most cases
    - Overlapping issues can be handled with extensions
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    if n_pred == 0 or n_gt == 0:
        return {
            'num_matches': 0,
            'quality_scores': [],
            'mean_quality': 0.0,
            'matched_ious': []
        }
    
    # Compute IoU matrix
    ious = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            ious[i, j] = compute_iou(pred, gt)
    
    # Greedy matching: sort predictions by best IoU
    matched_gt = set()
    quality_scores = []
    matched_ious = []
    
    # Get best IoU for each prediction
    best_ious = np.max(ious, axis=1)
    sorted_preds = np.argsort(-best_ious)  # Sort by best IoU (descending)
    
    for pred_idx in sorted_preds:
        # Find available GT boxes
        available_gt = [j for j in range(n_gt) if j not in matched_gt]
        
        if not available_gt:
            break
        
        # Find best available GT for this prediction
        best_gt = max(available_gt, key=lambda j: ious[pred_idx, j])
        best_iou = ious[pred_idx, best_gt]
        
        # Match if IoU meets threshold
        if best_iou >= min_iou:
            matched_gt.add(best_gt)
            quality_scores.append(iou_to_quality(best_iou, min_iou))
            matched_ious.append(best_iou)
    
    return {
        'num_matches': len(quality_scores),
        'quality_scores': quality_scores,
        'mean_quality': np.mean(quality_scores) if quality_scores else 0.0,
        'matched_ious': matched_ious
    }


# main F-beta reward computation

def compute_fbeta_reward(pred_boxes: List[List[float]], 
                         gt_boxes: List[List[float]],
                         beta: float = BETA,
                         min_iou: float = MIN_IOU) -> Dict[str, Any]:
    """
    Compute weighted F-beta reward using quality scores.
    
    Formula:
    1. Match boxes using greedy algorithm
    2. Compute quality-weighted TP = sum(quality_scores)
    3. Precision = weighted_TP / n_pred
    4. Recall = weighted_TP / n_gt  
    5. F_beta = (1 + β²) × P × R / (β² × P + R)
    
    This gives smooth, continuous rewards with partial credit.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    # Edge case 1: True negative (both empty)
    if n_pred == 0 and n_gt == 0:
        return {
            'reward': NO_FINDING_BONUS,
            'precision': 1.0,
            'recall': 1.0,
            'f_beta': 1.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'mean_quality': 1.0,
            'case': 'true_negative'
        }
    
    # Edge case 2: Complete miss or hallucination
    if n_pred == 0 or n_gt == 0:
        return {
            'reward': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f_beta': 0.0,
            'tp': 0,
            'fp': n_pred,
            'fn': n_gt,
            'mean_quality': 0.0,
            'case': 'hallucination' if n_pred > 0 else 'complete_miss'
        }
    
    # Main case: Perform greedy matching
    match_result = greedy_match(pred_boxes, gt_boxes, min_iou)
    
    num_matches = match_result['num_matches']
    quality_scores = match_result['quality_scores']
    mean_quality = match_result['mean_quality']
    
    # No matches found
    if num_matches == 0:
        return {
            'reward': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f_beta': 0.0,
            'tp': 0,
            'fp': n_pred,
            'fn': n_gt,
            'mean_quality': 0.0,
            'case': 'no_matches'
        }
    
    # Compute quality-weighted metrics
    weighted_tp = sum(quality_scores)
    
    # Soft precision and recall
    precision = weighted_tp / n_pred
    recall = weighted_tp / n_gt
    
    # F-beta score
    if precision + recall == 0:
        f_beta = 0.0
    else:
        beta_sq = beta * beta
        f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    return {
        'reward': f_beta,
        'precision': precision,
        'recall': recall,
        'f_beta': f_beta,
        'tp': num_matches,
        'fp': n_pred - num_matches,
        'fn': n_gt - num_matches,
        'mean_quality': mean_quality,
        'matched_ious': match_result['matched_ious'],
        'case': 'standard'
    }


# for VeRL RewardManager

def compute_score(data_source: str,
                 solution_str: str,
                 ground_truth: str,
                 extra_info: Optional[Dict[str, Any]] = None) -> float:
    """
    Main entry point for VeRL RewardManager.
    
    This SIMPLE base version:
    - Uses greedy matching (intuitive)
    - Computes weighted F-beta score
    - Returns reward in [0, 1]
    
    Extensions can add (in order of priority):
    1. Label matching bonus
    2. Difficulty weighting
    3. Safety thresholds
    4. Better overlapping handling
    
    Args:
        data_source: Dataset name
        solution_str: Model output
        ground_truth: Ground truth
        extra_info: Optional metadata
        
    Returns:
        Reward score in [0, 1]
    """
    # Extract boxes
    pred_boxes = extract_bounding_boxes(solution_str)
    gt_boxes = extract_bounding_boxes(ground_truth)
    
    # Compute base reward
    result = compute_fbeta_reward(pred_boxes, gt_boxes, beta=BETA, min_iou=MIN_IOU)
    base_reward = result['reward']
    
    # EXTENSION POINT: Add bonuses here 
    # Extension 2: Difficulty weighting 
    # if extra_info and 'difficulty_level' in extra_info:
    #     weights = {'easy': 0.85, 'medium': 1.0, 'hard': 1.2}
    #     base_reward *= weights.get(extra_info['difficulty_level'], 1.0)
    
    return float(np.clip(base_reward, 0.0, 1.0))

def analyze_prediction(solution_str: str,
                       ground_truth: str,
                       extra_info: Optional[Dict] = None) -> Dict[str, Any]:
    pred_boxes = extract_bounding_boxes(solution_str)
    gt_boxes = extract_bounding_boxes(ground_truth)
    
    result = compute_fbeta_reward(pred_boxes, gt_boxes)
    result['pred_boxes'] = pred_boxes
    result['gt_boxes'] = gt_boxes
    result['num_pred'] = len(pred_boxes)
    result['num_gt'] = len(gt_boxes)
    
    return result


def test_simple_base():
    """Test the simple base reward."""
    print("="*80)
    print("SIMPLE BASE REWARD TEST")
    print("="*80)
    print(f"\nHyperparameters (only 3!):")
    print(f"  MIN_IOU: {MIN_IOU}")
    print(f"  BETA: {BETA}")
    print(f"  NO_FINDING_BONUS: {NO_FINDING_BONUS}")
    
    test_cases = [
        {
            'name': '1. Perfect Match',
            'pred': '<answer>[[0.1, 0.1, 0.3, 0.3]]</answer>',
            'gt': '[[0.1, 0.1, 0.3, 0.3]]'
        },
        {
            'name': '2. True Negative',
            'pred': '<answer>No abnormalities</answer>',
            'gt': 'Clear chest X-ray'
        },
        {
            'name': '3. High IoU (0.8)',
            'pred': '<answer>[[0.11, 0.11, 0.31, 0.31]]</answer>',
            'gt': '[[0.1, 0.1, 0.3, 0.3]]'
        },
        {
            'name': '4. Medium IoU (0.55)',
            'pred': '<answer>[[0.12, 0.13, 0.32, 0.33]]</answer>',
            'gt': '[[0.1, 0.1, 0.3, 0.3]]'
        },
        {
            'name': '5. Low IoU (0.3)',
            'pred': '<answer>[[0.15, 0.15, 0.35, 0.35]]</answer>',
            'gt': '[[0.1, 0.1, 0.3, 0.3]]'
        },
        {
            'name': '6. Below Threshold (IoU=0.15)',
            'pred': '<answer>[[0.18, 0.18, 0.38, 0.38]]</answer>',
            'gt': '[[0.1, 0.1, 0.3, 0.3]]'
        },
        {
            'name': '7. Partial Detection (1 of 2)',
            'pred': '<answer>[[0.1, 0.1, 0.2, 0.2]]</answer>',
            'gt': '[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]]'
        },
        {
            'name': '8. Multiple Boxes (3 perfect)',
            'pred': '<answer>[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]</answer>',
            'gt': '[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]'
        },
        {
            'name': '9. Hallucination (1 FP)',
            'pred': '<answer>[[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.8, 0.8]]</answer>',
            'gt': '[[0.1, 0.1, 0.2, 0.2]]'
        },
        {
            'name': '10. Complete Miss',
            'pred': '<answer>No findings</answer>',
            'gt': '[[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]]'
        }
    ]
    
    print(f"{'Test Case':<30} {'N_pred':<8} {'N_gt':<8} {'Reward':<10} {'P':<8} {'R':<8} {'Case'}")
    print("-"*80)
    
    for test in test_cases:
        result = analyze_prediction(test['pred'], test['gt'])
        
        print(f"{test['name']:<30} {result['num_pred']:<8} {result['num_gt']:<8} "
              f"{result['reward']:<10.3f} {result['precision']:<8.3f} "
              f"{result['recall']:<8.3f} {result['case']}")
        
        # Show IoUs for matched boxes
        if result.get('matched_ious'):
            ious_str = ', '.join([f"{iou:.2f}" for iou in result['matched_ious']])
            print(f"  → Matched IoUs: [{ious_str}]")
    
    test_ious = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for iou in test_ious:
        quality = iou_to_quality(iou)
        if iou < MIN_IOU:
            note = " (below threshold → 0)"
        elif iou < 0.5:
            note = " (quadratic scaling)"
        else:
            note = " (linear scaling)"
        print(f"IoU = {iou:.1f} → Quality = {quality:.3f}{note}")
if __name__ == "__main__":
    #test_simple_base()
    pass