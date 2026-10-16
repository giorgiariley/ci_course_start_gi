import numpy as np

import ci_course


def test_summarises_a_complete_column():
    summary = ci_course.summarise(np.array([1.0, 2.0, 3.0]))

    assert summary["n"] == 3
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_missing_values_are_ignored():
    summary = ci_course.summarise(np.array([1.0, np.nan, 3.0]))

    assert summary["n"] == 2
    assert summary["mean"] == 2.0


def test_summarises_every_column():
    summaries = ci_course.summarise_all(ci_course.load_measurements())

    assert set(summaries) == {"temperature_c", "mass_g"}
    assert summaries["temperature_c"]["n"] == 7


def test_counts_missing_values():
    assert ci_course.count_missing(np.array([1.0, np.nan, 3.0, np.nan])) == 2
    assert ci_course.count_missing(np.array([1.0, 2.0])) == 0

def test_present_zero():
    summary =  ci_course.summarise(np.array([np.nan, np.nan, np.nan]))

    assert summary["n"] == 0


def test_non_number(tmp_path):
    csv = tmp_path/ "bad.csv"
    csv.write_test('sample_id, height_m\nA, 1.5\nB, non-a-number\n', encoding ="utf-8")

    errors = []
    data = ci_course.load_measurements(csv, errors=errors)

    assert np.isnan(data["height_m"][1])
    assert len(errors) == 1
