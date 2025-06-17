

def auto_split_reps(rep_indices: list[int], split_ratio: float = 0.8) -> tuple[list[int], list[int]]:
    n = len(rep_indices)
    if n == 1:
        return rep_indices, rep_indices
    elif n == 2:
        return [rep_indices[0]], [rep_indices[1]]
    elif n == 3:
        return rep_indices[:2], [rep_indices[2]]
    else:
        split_idx = int(round(n * split_ratio))
        return rep_indices[:split_idx], rep_indices[split_idx:]
