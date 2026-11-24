from datetime import timedelta


def in_window(measurement, t0, intime):
    return max(intime, t0 - timedelta(hours=24)) <= measurement < t0


def clip_interval(start, end, t0, intime):
    left = max(start, intime, t0 - timedelta(hours=24))
    right = min(end, t0)
    return (left, right) if left < right else None


def usable_urine(hours, period_start, window_start):
    return hours is not None and 24 <= hours <= 30 and (period_start >= window_start)


def total_score(scores, missing_zero=False):
    if not missing_zero and any(x is None for x in scores):
        return None
    return sum(0 if x is None else x for x in scores)
