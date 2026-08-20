"""Generate deterministic synthetic learning-event data for the public demo.

The output intentionally resembles the private application's scale while
containing no source rows, names, identifiers, or timestamps from real data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "public_data" / "learning_events.csv.gz"
DEFAULT_METADATA = ROOT / "public_data" / "metadata.json"
SEED = 20260820

# The public dataset mirrors the scale of the original 2022 demo subset.
COURSE_SIZES = {"DEMO-A": 60, "DEMO-B": 90, "DEMO-C": 90}
WEEKLY_ACTIVE_STUDENTS = {1: 228, 2: 214, 3: 198, 4: 180, 5: 182, 6: 181, 7: 177, 8: 88}
WEEKLY_EVENT_COUNTS = {1: 36_800, 2: 34_100, 3: 27_900, 4: 18_200, 5: 23_700, 6: 29_600, 7: 24_500, 8: 10_700}
ANOMALOUS_STUDENT_WEEKS = 72

OPERATIONS = np.array(
    [
        "NEXT",
        "PREV",
        "ADD_MARKER",
        "OPEN",
        "CLOSE",
        "PAGE_JUMP",
        "GETIT",
        "DELETE_MARKER",
        "ADD_BOOKMARK",
        "ADD_MEMO",
        "MEMO_TEXT_CHANGE_HISTORY",
        "BOOKMARK_JUMP",
        "NOTGETIT",
        "CHANGE_MEMO",
        "DELETE_BOOKMARK",
        "SEARCH_JUMP",
        "DELETE_MEMO",
        "ADD_HW_MEMO",
        "SEARCH",
        "LINK_CLICK",
        "CLEAR_HW_MEMO",
        "TIMER_STOP",
    ],
    dtype=object,
)

# Based on broad interaction patterns only; values are not copied rows.
BASE_WEIGHTS = np.array(
    [
        0.635,
        0.258,
        0.034,
        0.022,
        0.016,
        0.013,
        0.009,
        0.004,
        0.003,
        0.002,
        0.0015,
        0.0010,
        0.0008,
        0.0006,
        0.0003,
        0.0003,
        0.0002,
        0.0002,
        0.0001,
        0.0001,
        0.00005,
        0.00005,
    ],
    dtype=float,
)

PROFILE_PROBABILITIES = {
    "balanced": 0.46,
    "navigator": 0.25,
    "annotator": 0.14,
    "explorer": 0.09,
    "intensive": 0.06,
}


@dataclass(frozen=True)
class Student:
    course_id: str
    userid: str
    profile: str
    grade: str
    engagement: float
    device: str


def allocate_counts(total: int, weights: np.ndarray, minimum: int = 8) -> np.ndarray:
    """Allocate an exact event total across active students."""
    if total < len(weights) * minimum:
        raise ValueError("Total is too small for the requested minimum")
    remaining = total - len(weights) * minimum
    normalized = weights / weights.sum()
    raw = normalized * remaining
    counts = np.floor(raw).astype(int) + minimum
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - np.floor(raw)))
        counts[order[:remainder]] += 1
    return counts


def make_students(rng: np.random.Generator) -> list[Student]:
    profile_names = np.array(list(PROFILE_PROBABILITIES), dtype=object)
    profile_probs = np.array(list(PROFILE_PROBABILITIES.values()), dtype=float)
    students: list[Student] = []
    for course_id, size in COURSE_SIZES.items():
        for index in range(1, size + 1):
            profile = str(rng.choice(profile_names, p=profile_probs))
            profile_boost = {
                "balanced": 0.00,
                "navigator": -0.04,
                "annotator": 0.12,
                "explorer": 0.04,
                "intensive": 0.20,
            }[profile]
            engagement = float(np.clip(rng.normal(0.58 + profile_boost, 0.17), 0.10, 0.98))
            grade_score = engagement + float(rng.normal(0, 0.12))
            if grade_score >= 0.76:
                grade = "A"
            elif grade_score >= 0.61:
                grade = "B"
            elif grade_score >= 0.46:
                grade = "C"
            elif grade_score >= 0.31:
                grade = "D"
            else:
                grade = "F"
            device = str(rng.choice(["pc", "mobile", "tablet"], p=[0.72, 0.21, 0.07]))
            students.append(
                Student(
                    course_id=course_id,
                    userid=f"{course_id}_S{index:03d}",
                    profile=profile,
                    grade=grade,
                    engagement=engagement,
                    device=device,
                )
            )
    return students


def choose_active_students(
    rng: np.random.Generator, students: list[Student]
) -> dict[int, list[Student]]:
    engagement = np.array([student.engagement for student in students], dtype=float)
    active: dict[int, list[Student]] = {}
    for week, target in WEEKLY_ACTIVE_STUDENTS.items():
        # Higher-engagement synthetic students are slightly more likely to
        # remain active; random noise prevents a deterministic grade cutoff.
        persistence = engagement * rng.uniform(0.72, 1.28, len(students))
        probabilities = persistence / persistence.sum()
        indices = rng.choice(len(students), size=target, replace=False, p=probabilities)
        active[week] = [students[int(index)] for index in sorted(indices)]
    return active


def operation_probabilities(profile: str, anomaly: str) -> np.ndarray:
    weights = BASE_WEIGHTS.copy()
    index = {name: idx for idx, name in enumerate(OPERATIONS)}

    if profile == "navigator":
        weights[index["NEXT"]] *= 1.18
        weights[index["PREV"]] *= 1.12
    elif profile == "annotator":
        for name in ["ADD_MARKER", "DELETE_MARKER", "ADD_MEMO", "CHANGE_MEMO", "GETIT", "NOTGETIT"]:
            weights[index[name]] *= 5.0
    elif profile == "explorer":
        for name in ["PAGE_JUMP", "SEARCH", "SEARCH_JUMP", "LINK_CLICK", "BOOKMARK_JUMP"]:
            weights[index[name]] *= 6.0
    elif profile == "intensive":
        for name in ["ADD_MARKER", "ADD_BOOKMARK", "ADD_MEMO", "GETIT"]:
            weights[index[name]] *= 3.0

    if anomaly == "navigation_burst":
        weights *= 0.08
        weights[index["NEXT"]] = 0.78
        weights[index["PREV"]] = 0.15
        weights[index["PAGE_JUMP"]] = 0.05
    elif anomaly == "repeated_marker":
        weights *= 0.12
        weights[index["ADD_MARKER"]] = 0.68
        weights[index["DELETE_MARKER"]] = 0.18
        weights[index["NEXT"]] = 0.10
    elif anomaly == "repeated_jump":
        weights *= 0.10
        weights[index["PAGE_JUMP"]] = 0.72
        weights[index["NEXT"]] = 0.13
        weights[index["PREV"]] = 0.10

    return weights / weights.sum()


def write_student_week(
    writer: csv.DictWriter,
    rng: np.random.Generator,
    student: Student,
    week: int,
    count: int,
    anomaly: str,
) -> None:
    probabilities = operation_probabilities(student.profile, anomaly)
    middle_count = max(0, count - 2)
    middle = rng.choice(OPERATIONS, size=middle_count, p=probabilities).tolist()
    operations = ["OPEN", *middle, "CLOSE"] if count >= 2 else ["OPEN"]

    course_offset = {"DEMO-A": 0, "DEMO-B": 1, "DEMO-C": 2}[student.course_id]
    class_date = datetime(2025, 4, 7, 9, 0) + timedelta(days=(week - 1) * 7 + course_offset)
    start = class_date + timedelta(minutes=int(rng.integers(0, 45)), seconds=int(rng.integers(0, 60)))
    increments = rng.integers(1, 13, size=len(operations))
    elapsed = np.cumsum(increments)
    page = int(rng.integers(1, 6))
    content = f"{student.course_id}_W{week:02d}_M{int(rng.integers(1, 4)):02d}"
    memo_length = 0

    for op, seconds in zip(operations, elapsed):
        if op == "NEXT":
            page = min(80, page + 1)
        elif op == "PREV":
            page = max(1, page - 1)
        elif op in {"PAGE_JUMP", "SEARCH_JUMP", "BOOKMARK_JUMP"}:
            page = int(rng.integers(1, 81))

        marker = ""
        if op == "ADD_MARKER":
            marker = str(rng.choice(["yellow", "blue", "pink", "green"]))
        if op in {"ADD_MEMO", "CHANGE_MEMO", "MEMO_TEXT_CHANGE_HISTORY", "ADD_HW_MEMO"}:
            memo_length = max(1, memo_length + int(rng.integers(3, 45)))
        elif op == "DELETE_MEMO":
            memo_length = 0

        event_time = start + timedelta(seconds=int(seconds))
        writer.writerow(
            {
                "userid": student.userid,
                "course_id": student.course_id,
                "week": week,
                "contentsid": content,
                "operationname": op,
                "pageno": page,
                "marker": marker,
                "memo_length": memo_length,
                "devicecode": student.device,
                "eventtime": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                "grade": student.grade,
                "synthetic_profile": student.profile,
                "is_injected_anomaly": int(bool(anomaly)),
                "anomaly_scenario": anomaly,
            }
        )


def generate(output: Path, metadata_path: Path, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    students = make_students(rng)
    active_by_week = choose_active_students(rng, students)
    active_pairs = [(student.userid, week) for week, active in active_by_week.items() for student in active]
    chosen = rng.choice(len(active_pairs), size=ANOMALOUS_STUDENT_WEEKS, replace=False)
    anomaly_pairs = {active_pairs[int(index)] for index in chosen}
    scenario_names = ["navigation_burst", "very_low_activity", "repeated_marker", "repeated_jump"]
    anomaly_scenarios = {
        pair: scenario_names[index % len(scenario_names)]
        for index, pair in enumerate(sorted(anomaly_pairs))
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "userid",
        "course_id",
        "week",
        "contentsid",
        "operationname",
        "pageno",
        "marker",
        "memo_length",
        "devicecode",
        "eventtime",
        "grade",
        "synthetic_profile",
        "is_injected_anomaly",
        "anomaly_scenario",
    ]

    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for week, active in active_by_week.items():
            weights = []
            scenarios = []
            for student in active:
                scenario = anomaly_scenarios.get((student.userid, week), "")
                profile_multiplier = {
                    "balanced": 1.00,
                    "navigator": 1.05,
                    "annotator": 0.92,
                    "explorer": 0.90,
                    "intensive": 1.35,
                }[student.profile]
                anomaly_multiplier = {
                    "": 1.0,
                    "navigation_burst": 3.0,
                    "very_low_activity": 0.08,
                    "repeated_marker": 1.4,
                    "repeated_jump": 1.5,
                }[scenario]
                weights.append(
                    max(
                        0.01,
                        student.engagement
                        * profile_multiplier
                        * anomaly_multiplier
                        * float(rng.lognormal(mean=0.0, sigma=0.42)),
                    )
                )
                scenarios.append(scenario)
            counts = allocate_counts(WEEKLY_EVENT_COUNTS[week], np.asarray(weights), minimum=8)
            for student, count, scenario in zip(active, counts, scenarios):
                write_student_week(writer, rng, student, week, int(count), scenario)

    metadata = {
        "dataset": "Synthetic Learning Events for Public Demo",
        "synthetic": True,
        "seed": seed,
        "students": len(students),
        "courses": COURSE_SIZES,
        "weeks": list(WEEKLY_ACTIVE_STUDENTS),
        "student_week_records": sum(WEEKLY_ACTIVE_STUDENTS.values()),
        "events": sum(WEEKLY_EVENT_COUNTS.values()),
        "weekly_active_students": WEEKLY_ACTIVE_STUDENTS,
        "weekly_event_counts": WEEKLY_EVENT_COUNTS,
        "injected_anomalous_student_weeks": ANOMALOUS_STUDENT_WEEKS,
        "contains_real_personal_data": False,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    metadata = generate(args.output, args.metadata, args.seed)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
