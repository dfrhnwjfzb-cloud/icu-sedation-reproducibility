import math


def missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def score(var, x, vent=False):
    if missing(x):
        return None
    if var == "pf":
        return (
            4
            if vent and x < 100
            else 3
            if vent and x < 200
            else 2
            if x < 300
            else 1
            if x < 400
            else 0
        )
    if var == "platelet":
        return 4 if x < 20 else 3 if x < 50 else 2 if x < 100 else 1 if x < 150 else 0
    if var == "bilirubin":
        return 4 if x >= 12 else 3 if x >= 6 else 2 if x >= 2 else 1 if x >= 1.2 else 0
    if var == "gcs":
        return 4 if x < 6 else 3 if x <= 9 else 2 if x <= 12 else 1 if x <= 14 else 0
    if var == "creatinine":
        return 4 if x >= 5 else 3 if x >= 3.5 else 2 if x >= 2 else 1 if x >= 1.2 else 0
    if var == "urine24":
        return 4 if x < 200 else 3 if x < 500 else 0
    if var == "map":
        return int(x < 70)
    if var in ["epinephrine", "norepinephrine"]:
        return 4 if x > 0.1 else 3 if x > 0 else 0
    if var == "dopamine":
        return 4 if x > 15 else 3 if x > 5 else 2 if x > 0 else 0
    if var == "dobutamine":
        return 2 if x > 0 else 0
    raise ValueError(var)


def apache2_gate(components, age, chronic_points, acute_renal_failure):
    if (
        chronic_points is None
        or acute_renal_failure is None
        or any(v is None for v in components.values())
    ):
        raise ValueError(
            "APACHE II infeasible: unknown critical inputs cannot be set to normal"
        )
    raise NotImplementedError(
        "A complete APACHE II implementation is intentionally gated pending valid chronic-health and renal evidence"
    )
