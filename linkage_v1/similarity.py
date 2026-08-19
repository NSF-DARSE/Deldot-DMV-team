from __future__ import annotations


def jaro_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0 if left else 0.0
    if not left or not right:
        return 0.0

    match_distance = max(len(left), len(right)) // 2 - 1
    match_distance = max(0, match_distance)
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)

    matches = 0
    for i, left_char in enumerate(left):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(right))
        for j in range(start, end):
            if right_matches[j] or left_char != right[j]:
                continue
            left_matches[i] = True
            right_matches[j] = True
            matches += 1
            break

    if not matches:
        return 0.0

    left_order = [left[i] for i, matched in enumerate(left_matches) if matched]
    right_order = [right[j] for j, matched in enumerate(right_matches) if matched]
    transpositions = sum(a != b for a, b in zip(left_order, right_order)) / 2.0

    return (
        matches / len(left)
        + matches / len(right)
        + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(left: str, right: str, prefix_scale: float = 0.1) -> float:
    base = jaro_similarity(left, right)
    if base <= 0.7:
        return base
    prefix = 0
    for a, b in zip(left[:4], right[:4]):
        if a != b:
            break
        prefix += 1
    return min(1.0, base + prefix * prefix_scale * (1.0 - base))
