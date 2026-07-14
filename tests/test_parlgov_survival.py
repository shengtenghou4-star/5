from datetime import date

import pytest

from fencha.datasets.parlgov import CabinetRecord
from fencha.datasets.parlgov_survival import (
    M3_BUILDER_VERSION,
    build_leader_survival_cases,
    normalize_horizons,
)


def _cabinet(
    cabinet_id: str,
    leader_name: str,
    start_date: date,
) -> CabinetRecord:
    return CabinetRecord(
        cabinet_id=cabinet_id,
        country_code="TST",
        country_name="Testland",
        cabinet_name=leader_name,
        leader_name=leader_name,
        start_date=start_date,
        election_date=date(2019, 12, 1),
        government_seats=48,
        parliament_seats=100,
        coalition_size=2,
        cabinet_type="minority",
    )


def test_multihorizon_cases_have_correct_labels_and_unique_domains() -> None:
    cases = build_leader_survival_cases(
        [
            _cabinet("a", "Leader A", date(2020, 1, 1)),
            _cabinet("b", "Leader B", date(2020, 7, 1)),
        ],
        as_of=date(2021, 12, 31),
        horizons=(180, 30, 90, 90),
        earliest_cutoff=date(2020, 1, 1),
    )

    target = {
        case.domain: case
        for case in cases
        if case.case_id.startswith("parlgov:TST:Leader A:2020-05-01:")
    }
    assert set(target) == {
        "government_leader_exit_30d",
        "government_leader_exit_90d",
        "government_leader_exit_180d",
    }
    assert target["government_leader_exit_30d"].outcome is False
    assert target["government_leader_exit_90d"].outcome is True
    assert target["government_leader_exit_180d"].outcome is True

    for horizon in (30, 90, 180):
        case = target[f"government_leader_exit_{horizon}d"]
        assert case.case_id.endswith(f":{horizon}d")
        assert f"within {horizon} days?" in case.question
        assert case.tags == (
            "TST",
            "parlgov",
            M3_BUILDER_VERSION,
            f"horizon:{horizon}",
        )
        assert all(
            feature.observed_at <= case.cutoff_at for feature in case.features.values()
        )


def test_normalize_horizons_rejects_empty_and_nonpositive_values() -> None:
    assert normalize_horizons((365, 30, 90, 30)) == (30, 90, 365)
    with pytest.raises(ValueError, match="at least one"):
        normalize_horizons(())
    with pytest.raises(ValueError, match="positive"):
        normalize_horizons((0, 30))
