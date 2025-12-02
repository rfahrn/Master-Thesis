"""
R3: Multi-Objective Reward Function with Hungarian Matching
"""
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional
PENALTY_FLOOR = -0.45          # Minimum reward for very poor predictions
NO_FINDING_REWARD = 0.20        # Reward for correctly predicting no abnormalities
# penalty weights
W_SIZE = 0.20                   # Weight for size penalty
W_ASPECT = 0.10                 # Weight for aspect ratio penalty  
W_CENTER = 0.10                 # Weight for center distance penalty
# F-beta parameter
BETA = 1.5                      # Emphasizes recall over precision
# Over-prediction penalties
FP_TOLERANCE = 1.3              # Allow up to 1.3× GT boxes before penalizing
LAMBDA_SPAM = 0.6               # Strength of over-prediction penalty
# False positive penalties (when GT = 0)
BETA_FP_BASE = 0.25             # Base false positive penalty
LAMBDA_FP = 0.5                 # Decay rate for multiple FPs
GAMMA_FP = 1.5                  # Superlinear decay exponent
def piecewise_linear_reward(iou: float) -> float:
    iou = float(np.clip(iou, 0.0, 1.0))
    
    if iou <= 0.20:
        # Strong penalty → slight penalty
        # Gradient = 2.0 (steep recovery)
        reward = -0.50 + 2.0 * iou
    elif iou <= 0.50:
        # Enter positive rewards
        # Gradient = 1.0 (steady improvement)
        reward = -0.10 + 1.0 * (iou - 0.20)
    elif iou <= 0.85:
        # Good zone
        # Gradient = 1.0 (maintain signal)
        reward = 0.20 + 1.0 * (iou - 0.50)
    else:
        reward = 0.55 + 3.0 * (iou - 0.85)
    if reward < 0:
        reward = reward * (PENALTY_FLOOR / -0.50)
    
    return float(np.clip(reward, PENALTY_FLOOR, 1.0))


def smooth_exponential_reward(iou: float) -> float:
    iou = float(np.clip(iou, 0.0, 1.0))
    reward = -0.50 + 1.50 * (1.0 - np.exp(-3.0 * iou))
    if reward < 0:
        reward = reward * (PENALTY_FLOOR / -0.50)
    return float(np.clip(reward, PENALTY_FLOOR, 1.0))
   
base_reward_function = piecewise_linear_reward  
#base_reward_function = smooth_exponential_reward 

def extract_bounding_boxes(answer: str) -> List[List[float]]:
    """Extract [x1, y1, x2, y2] bounding boxes from text."""
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"

    boxes = []
    for match in re.finditer(pattern, answer):
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            if not all(np.isfinite(coords)):
                continue

            x1, y1, x2, y2 = coords
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if (x2 - x1) > 0 and (y2 - y1) > 0:
                boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue

    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ============================================================================
# MULTI-OBJECTIVE PENALTIES
# ============================================================================

def compute_size_penalty(pred_box: List[float], gt_box: List[float]) -> float:
    """
    Logarithmic size penalty.
    Penalizes boxes that are too large or too small relative to ground truth.
    """
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    if gt_area == 0 or pred_area == 0:
        return 1.0

    penalty = abs(np.log(pred_area / gt_area)) / 3.0
    return float(np.clip(penalty, 0.0, 1.0))


def compute_aspect_ratio_penalty(pred_box: List[float], gt_box: List[float]) -> float:
    """
    Aspect ratio penalty.
    Penalizes boxes with incorrect shape (too wide or too tall).
    Normalized by max aspect ratio for scale invariance.
    """
    pred_w = pred_box[2] - pred_box[0]
    pred_h = pred_box[3] - pred_box[1]
    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]

    if pred_h == 0 or gt_h == 0:
        return 0.0

    pred_ar = pred_w / pred_h
    gt_ar = gt_w / gt_h

    diff = abs(pred_ar - gt_ar) / max(pred_ar, gt_ar)
    return float(np.clip(diff, 0.0, 1.0))


def compute_center_distance_penalty(pred_box: List[float], gt_box: List[float]) -> float:
    """
    Center distance penalty.
    Penalizes boxes that are offset from the correct location.
    Normalized by diagonal of GT box for scale invariance.
    """
    pred_cx = (pred_box[0] + pred_box[2]) / 2
    pred_cy = (pred_box[1] + pred_box[3]) / 2
    gt_cx = (gt_box[0] + gt_box[2]) / 2
    gt_cy = (gt_box[1] + gt_box[3]) / 2

    # Euclidean distance between centers
    dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)

    # Normalize by diagonal
    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]
    diagonal = np.sqrt(gt_w**2 + gt_h**2)

    if diagonal == 0:
        return 0.0

    penalty = dist / diagonal
    return float(np.clip(penalty, 0.0, 1.0))

def hungarian_match(iou_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Hungarian (Kuhn-Munkres) optimal bipartite matching algorithm.
    
    Finds the globally optimal one-to-one assignment that maximizes
    the total IoU across all matched pairs.
    
    Algorithm:
    1. Convert IoU matrix to cost matrix: Cost = 1 - IoU
    2. Apply Hungarian algorithm via scipy.optimize.linear_sum_assignment
    3. Extract matches and filter out zero-IoU assignments
    
    Time complexity: O(max(M,N)³)
    Space complexity: O(M × N)
    
    Advantages over Greedy:
    - Globally optimal assignment (maximizes total IoU)
    - Better handling of overlapping/ambiguous boxes
    - Consistent with DETR-style object detection training
    - Theoretically grounded (solves assignment problem exactly)
    
    Thesis explanation:
    "We employ the Hungarian algorithm to find the optimal bipartite matching
    between predicted and ground truth boxes. This ensures globally optimal
    assignment that maximizes total IoU, avoiding suboptimal local decisions
    that greedy matching might produce with overlapping boxes."
    """
    M, N = iou_matrix.shape  # M predictions, N ground truth
    
    if M == 0 or N == 0:
        return [], []
    
    # Convert to cost matrix (Hungarian minimizes cost, we want to maximize IoU)
    # Use 1 - IoU as cost, so higher IoU = lower cost
    cost_matrix = 1.0 - iou_matrix
    
    # Solve the assignment problem
    # linear_sum_assignment returns (row_indices, col_indices) for optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Extract matches and their IoU values
    # Filter out matches with IoU = 0 (no overlap)
    matches = []
    iou_values = []
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        if iou > 0:  # Only keep matches with actual overlap
            matches.append((int(pred_idx), int(gt_idx)))
            iou_values.append(float(iou))
    
    return matches, iou_values


# ============================================================================
# F-BETA SCORE
# ============================================================================

def compute_fbeta_score(tp_weighted: float, num_pred: int, num_gt: int,
                       beta: float = BETA) -> float:
    """
    Compute F-beta score with weighted true positives.
    
    Standard F-beta formula:
        F_β = (1 + β²) × (precision × recall) / (β² × precision + recall)
    
    Where:
        precision = Σ r_adjusted / M  (sum of adjusted rewards / num predictions)
        recall = Σ r_adjusted / N     (sum of adjusted rewards / num ground truth)
    
    β = 1.5 emphasizes recall slightly over precision 
    """
    eps = 1e-8

    if num_gt == 0:
        return NO_FINDING_REWARD if num_pred == 0 else 0.0

    if num_pred == 0:
        return 0.0

    precision = tp_weighted / (num_pred + eps)
    recall = tp_weighted / (num_gt + eps)

    beta_sq = beta * beta
    fbeta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall + eps)

    return float(np.clip(fbeta, 0.0, 1.0))

def compute_reward(predicted_boxes: List[List[float]],
                  actual_boxes: List[List[float]],
                  beta: float = BETA) -> Tuple[float, Dict]:
    """
    Compute reward with Hungarian matching and multi-objective penalties.
    
    Algorithm:
    1. Build IoU matrix between all predicted and ground truth boxes
    2. Hungarian match: find optimal assignment maximizing total IoU
    3. For each match (i, j):
       a. Compute base reward: r_base = f(IoU)
       b. Compute penalties: size, aspect ratio, center distance
       c. Adjust reward: r_adj = r_base × (1 - weighted_penalties)
    4. Compute F-beta score from sum of adjusted rewards
    5. Apply over-prediction penalty if too many predictions
    6. Apply perfect recall bonus if all boxes well-matched
    
    Returns:
        score: Final reward ∈ [PENALTY_FLOOR, 1.0]
        diagnostics: Detailed breakdown for logging
    """
    M = len(predicted_boxes)
    N = len(actual_boxes)

    diagnostics = {
        'num_predicted': M,
        'num_ground_truth': N,
        'matched_boxes': 0,
        'avg_iou': 0.0,
        'avg_base_reward': 0.0,
        'avg_size_penalty': 0.0,
        'avg_aspect_penalty': 0.0,
        'avg_center_penalty': 0.0,
        'fbeta_score': 0.0,
        'spam_penalty_applied': 0.0,
    }

    # Case 1: No ground truth (no abnormalities)
    if N == 0:
        if M == 0:
            # Correct "no finding"
            return NO_FINDING_REWARD, diagnostics
        else:
            # False positives
            fp_penalty = BETA_FP_BASE * np.exp(-LAMBDA_FP * (M ** GAMMA_FP))
            diagnostics['fbeta_score'] = NO_FINDING_REWARD
            return -float(fp_penalty), diagnostics

    # Case 2: Missed all findings
    if M == 0:
        return 0.0, diagnostics

    # Case 3: Normal matching
    # Build IoU matrix
    iou_matrix = np.zeros((M, N))
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(actual_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)

    # Hungarian matching (optimal!)
    matches, iou_values = hungarian_match(iou_matrix)

    # Compute multi-objective adjusted rewards
    tp_weighted = 0.0
    penalties = {'size': [], 'aspect': [], 'center': []}
    base_rewards = []

    for (pred_idx, gt_idx), iou in zip(matches, iou_values):
        # 1. Base reward from IoU
        r_base = base_reward_function(iou)
        base_rewards.append(r_base)

        # 2. Multi-objective penalties
        p_size = compute_size_penalty(
            predicted_boxes[pred_idx],
            actual_boxes[gt_idx]
        )
        penalties['size'].append(p_size)

        p_aspect = compute_aspect_ratio_penalty(
            predicted_boxes[pred_idx],
            actual_boxes[gt_idx]
        )
        penalties['aspect'].append(p_aspect)

        p_center = compute_center_distance_penalty(
            predicted_boxes[pred_idx],
            actual_boxes[gt_idx]
        )
        penalties['center'].append(p_center)

        # 3. Combined penalty
        p_total = (W_SIZE * p_size +
                  W_ASPECT * p_aspect +
                  W_CENTER * p_center)

        # 4. Adjusted reward
        # Only apply penalties to positive rewards
        if r_base < 0:
            r_adjusted = r_base
        else:
            r_adjusted = r_base * (1.0 - p_total)

        tp_weighted += r_adjusted

    # Handle case where all matches are negative
    if tp_weighted < 0:
        penalty_score = tp_weighted / N
        diagnostics['avg_base_reward'] = np.mean(base_rewards) if base_rewards else 0.0
        return float(np.clip(penalty_score, PENALTY_FLOOR, 0.0)), diagnostics

    # F-beta score
    fbeta = compute_fbeta_score(tp_weighted, M, N, beta)

    # Over-prediction penalty (exponential decay)
    spam_penalty = 0.0
    if M > FP_TOLERANCE * N:
        excess = M - FP_TOLERANCE * N
        spam_penalty = LAMBDA_SPAM * excess
        fbeta *= np.exp(-spam_penalty)

    # Perfect recall bonus
    if N >= 2 and len(matches) == N:
        if all(iou >= 0.80 for iou in iou_values):
            recall_bonus = 0.08
            fbeta = min(1.0, fbeta + recall_bonus)

    # Diagnostics
    diagnostics['matched_boxes'] = len(matches)
    diagnostics['avg_iou'] = np.mean(iou_values) if iou_values else 0.0
    diagnostics['avg_base_reward'] = np.mean(base_rewards) if base_rewards else 0.0
    diagnostics['avg_size_penalty'] = np.mean(penalties['size']) if penalties['size'] else 0.0
    diagnostics['avg_aspect_penalty'] = np.mean(penalties['aspect']) if penalties['aspect'] else 0.0
    diagnostics['avg_center_penalty'] = np.mean(penalties['center']) if penalties['center'] else 0.0
    diagnostics['fbeta_score'] = fbeta
    diagnostics['spam_penalty_applied'] = spam_penalty

    return float(np.clip(fbeta, PENALTY_FLOOR, 1.0)), diagnostics


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                 extra_info: Optional[dict] = None) -> float:
    predicted_boxes = extract_bounding_boxes(solution_str)

    if extra_info and 'bounding_boxes' in extra_info:
        ground_truth_boxes = extra_info['bounding_boxes']
    else:
        ground_truth_boxes = extract_bounding_boxes(ground_truth)

    score, _ = compute_reward(predicted_boxes, ground_truth_boxes, BETA)

    return float(score)

def greedy_match(iou_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Greedy matching for comparison purposes.
    Kept here to demonstrate the difference with Hungarian.
    """
    M, N = iou_matrix.shape
    
    if M == 0 or N == 0:
        return [], []
    
    matches = []
    iou_values = []
    used_pred = set()
    
    for gt_idx in range(N):
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx in range(M):
            if pred_idx not in used_pred:
                current_iou = iou_matrix[pred_idx, gt_idx]
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_idx = pred_idx
        
        if best_pred_idx >= 0 and best_iou > 0:
            matches.append((best_pred_idx, gt_idx))
            iou_values.append(best_iou)
            used_pred.add(best_pred_idx)
    
    return matches, iou_values


def compare_matching_algorithms():
    """
    Demonstrate cases where Hungarian outperforms Greedy matching.
    """
    print("comparison greedy vs hungarian")
    test_cases = [
        {
            'name': 'Simple case (both methods equal)',
            'iou_matrix': np.array([
                [0.8, 0.1],  # Pred0: high with GT0, low with GT1
                [0.1, 0.7],  # Pred1: low with GT0, high with GT1
            ])
        },
        {
            'name': 'Ambiguous case (Hungarian wins)',
            'iou_matrix': np.array([
                [0.6, 0.5],  # Pred0: good with both
                [0.7, 0.3],  # Pred1: best with GT0
            ])
        },
        {
            'name': 'Three boxes (more complex)',
            'iou_matrix': np.array([
                [0.5, 0.4, 0.1],  # Pred0
                [0.6, 0.5, 0.2],  # Pred1
                [0.3, 0.7, 0.4],  # Pred2
            ])
        },
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}")
        print("-" * 50)
        iou_mat = case['iou_matrix']
        print(f"IoU Matrix:\n{iou_mat}")
        
        # Greedy matching
        greedy_matches, greedy_ious = greedy_match(iou_mat)
        greedy_total = sum(greedy_ious)
        
        # Hungarian matching
        hungarian_matches, hungarian_ious = hungarian_match(iou_mat)
        hungarian_total = sum(hungarian_ious)
        
        print(f"\nGreedy:    matches={greedy_matches}, total_iou={greedy_total:.3f}")
        print(f"Hungarian: matches={hungarian_matches}, total_iou={hungarian_total:.3f}")
        
        if hungarian_total > greedy_total + 0.001:
            print("→ Hungarian finds better assignment!")
        else:
            print("→ Both methods find same solution")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_matching_algorithms()
