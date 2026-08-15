import statistics
from dataclasses import dataclass


@dataclass
class BoxLike:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


def assign_shelves(
    detections: list[BoxLike],
    gap_factor: float = 0.6,
    min_gap_px: float = 4.0,
) -> tuple[list[int], int]:
    """Cluster detections into shelf rows by the bounding-box bottom edge (y2).

    Bottom edge is used rather than box center because products of different
    heights standing on the same physical shelf share roughly the same base
    y-coordinate, while their centers diverge with height.

    Shelf 1 = topmost row (smallest y2), numbering increases downward.
    Returns (shelf ids aligned to `detections` order, total shelf row count).
    """
    n = len(detections)
    if n == 0:
        return [], 0
    if n == 1:
        return [1], 1

    order = sorted(range(n), key=lambda i: detections[i].bbox[3])
    ys = [detections[i].bbox[3] for i in order]

    heights = [d.bbox[3] - d.bbox[1] for d in detections]
    median_height = statistics.median(heights)
    threshold = max(gap_factor * median_height, min_gap_px)

    shelf_of_sorted = [1] * n
    current_shelf = 1
    for i in range(1, n):
        gap = ys[i] - ys[i - 1]
        if gap > threshold:
            current_shelf += 1
        shelf_of_sorted[i] = current_shelf

    shelf_ids = [0] * n
    for sorted_pos, original_idx in enumerate(order):
        shelf_ids[original_idx] = shelf_of_sorted[sorted_pos]

    return shelf_ids, current_shelf
