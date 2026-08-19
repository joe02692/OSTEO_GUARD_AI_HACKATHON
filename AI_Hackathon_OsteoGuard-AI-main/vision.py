"""Knee X-ray analysis.

=============================================================================
 IMPORTANT - THERE IS NO TRAINED VISION MODEL IN THIS PROJECT YET.
=============================================================================

Nothing here measures anything from the image. `analyze_xray` returns
placeholder values so the dashboard layout can be built and demonstrated.
Every consumer of this module MUST surface `result["simulated"] == True`
to the user. Presenting these numbers as real clinical measurements would
be fabricating medical output.

To make this real, implement `_predict_with_model` below and flip
MODEL_AVAILABLE to True. A drop-in option is a Kellgren-Lawrence grading
CNN trained on a knee OA dataset (e.g. the OAI / Kaggle knee KL datasets);
load it once at import and return the same dict shape.
"""

import hashlib

MODEL_AVAILABLE = False
MODEL_PATH = None  # e.g. "models/kl_grader.pt" once a model exists

KL_DESCRIPTIONS = {
    0: "No radiographic features of osteoarthritis.",
    1: "Doubtful joint space narrowing, possible osteophytic lipping.",
    2: "Definite osteophytes, possible joint space narrowing.",
    3: "Moderate joint space narrowing, multiple osteophytes, possible deformity.",
    4: "Large osteophytes, marked joint space narrowing, severe sclerosis.",
}

SEVERITY_BY_GRADE = {0: "None", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}


def _predict_with_model(image_bytes):
    """Real inference goes here. Must return the same dict shape as
    `_placeholder_result`, with `simulated` set to False."""
    raise NotImplementedError(
        "No vision model is connected. Train or download a KL-grading model, "
        "set MODEL_PATH, implement this function, and set MODEL_AVAILABLE = True."
    )


def _placeholder_result(image_bytes):
    """Deterministic placeholder derived from the file's hash.

    Deterministic (not random) so the same upload gives a stable result across
    Streamlit reruns -- a demo that reshuffles its 'diagnosis' on every click is
    obviously broken. These are NOT measurements.
    """
    digest = hashlib.sha256(image_bytes).digest()

    grade = digest[0] % 5
    confidence = 78.0 + (digest[1] % 180) / 10.0          # 78.0 - 95.9
    joint_space = 1.4 + (digest[2] % 36) / 10.0           # 1.4 - 4.9 mm

    return {
        "simulated": True,
        "detected": grade >= 2,
        "kl_grade": grade,
        "kl_description": KL_DESCRIPTIONS[grade],
        "severity": SEVERITY_BY_GRADE[grade],
        "confidence": round(confidence, 1),
        "joint_space_mm": round(joint_space, 1),
        "joint_space_normal": "3-5 mm",
        "osteophytes": "Detected" if grade >= 2 else "Not detected",
        "sclerosis": ["None", "None", "Mild", "Mild", "Marked"][grade],
        "alignment": ["Normal", "Normal", "Slight varus", "Slight varus", "Varus"][grade],
    }


def analyze_xray(image_bytes):
    """Analyse a knee X-ray.

    Returns a dict describing the finding. Check `result["simulated"]` -- when
    True the values are placeholders and must be labelled as such in the UI.
    """
    if MODEL_AVAILABLE:
        return _predict_with_model(image_bytes)
    return _placeholder_result(image_bytes)


def risk_factors(age, bmi, previous_injury, family_history, physical_load):
    """Rule-based risk display derived from the patient inputs.

    These are simple, transparent thresholds from well-established OA risk
    factors -- not a trained model, but genuinely computed from what the user
    entered rather than invented.
    """

    def band(value, moderate, high):
        if value >= high:
            return "High", 0.9
        if value >= moderate:
            return "Moderate", 0.6
        return "Low", 0.3

    age_label, age_val = band(age, 50, 65)
    bmi_label, bmi_val = band(bmi, 25, 30)

    def yes_no(flag, high_label="Moderate", high_val=0.6):
        return (high_label, high_val) if flag else ("Low", 0.3)

    injury_label, injury_val = yes_no(previous_injury)
    family_label, family_val = yes_no(family_history)
    load_label, load_val = yes_no(physical_load)

    return [
        {"name": "Age (> 50)", "level": age_label, "value": age_val},
        {"name": "High BMI", "level": bmi_label, "value": bmi_val},
        {"name": "Previous Injury", "level": injury_label, "value": injury_val},
        {"name": "Family History", "level": family_label, "value": family_val},
        {"name": "High Physical Load", "level": load_label, "value": load_val},
    ]
