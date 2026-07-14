from datetime import date

from fencha.datasets.parlgov import CabinetRecord
from fencha.datasets.parlgov_competing import (
    M3_COMPETING_BUILDER_VERSION,
    OTHER_RECORDED_TRANSITION,
    POST_ELECTION_TRANSITION,
    build_leader_competing_risk_cases,
    classify_transition_mechanism,
)


def _cabinet(
    cabinet_id: str,
    leader_name: str,
    start_date: date,
    election_date: date | None,
) -> CabinetRecord:
    return CabinetRecord(
        cabinet_id=cabinet_id,
        country_code="TST",
        country_name="Testland",
        cabinet_name=leader_name,
        leader_name=leader_name,
        start_date=start_date,
        election_date=election_date,
        government_seats=48,
        parliament_seats=100,
        coalition_size=2,
        cabinet_type="minority",
    )


def test_builder_separates_election_linked_and_other_transitions() -> None:
    cases = build_leader_competing_risk_cases(
        [
            _cabinet("a", "Leader A", date(2020, 1, 1), date(2019, 1, 1)),
            _cabinet("b", "Leader B", date(2020, 7, 1), date(2020, 6, 1)),
            _cabinet("c", "Leader C", date(2021, 1, 1), date(2018, 1, 1)),
        ],
        as_of=date(2021, 12, 31),
        horizons=(90,),
        earliest_cutoff=date(2020, 1, 1),
    )

    leader_a = {
        case.domain: case
        for case in cases
        if ":Leader A:2020-05-01:90d:" in case.case_id
    }
    assert leader_a[
        "government_leader_exit_post_election_transition_90d"
    ].outcome is True
    assert leader_a[
        "government_leader_exit_other_recorded_transition_90d"
    ].outcome is False
    assert all("total_exit:1" in case.tags for case in leader_a.values())

    leader_b = {
        case.domain: case
        for case in cases
        if ":Leader B:2020-11-01:90d:" in case.case_id
    }
    assert leader_b[
        "government_leader_exit_post_election_transition_90d"
    ].outcome is False
    assert leader_b[
        "government_leader_exit_other_recorded_transition_90d"
    ].outcome is True
    assert all("total_exit:1" in case.tags for case in leader_b.values())

    leader_c = [
        case
        for case in cases
        if ":Leader C:" in case.case_id and case.cutoff_at.date() <= date(2021, 9, 1)
    ]
    assert leader_c
    assert all(case.outcome is False for case in leader_c)
    assert all("total_exit:0" in case.tags for case in leader_c)
    assert M3_COMPETING_BUILDER_VERSION in {tag for case in cases for tag in case.tags}

    assert all(case.resolved_at > case.cutoff_at for case in cases)
    assert all(
        feature.observed_at <= case.cutoff_at
        for case in cases
        for feature in case.features.values()
    )
    assert all(
        sum(tag.startswith("total_exit:") for tag in case.tags) == 1
        for case in cases
    )


def test_classifier_does_not_use_an_election_after_the_transition() -> None:
    next_cabinet = _cabinet(
        "b",
        "Leader B",
        date(2020, 7, 1),
        date(2020, 7, 10),
    )
    mechanism, election_date = classify_transition_mechanism(
        exit_date=date(2020, 7, 1),
        next_cabinet=next_cabinet,
    )
    assert mechanism == OTHER_RECORDED_TRANSITION
    assert election_date == date(2020, 7, 10)


def test_classifier_uses_recent_recorded_election_only() -> None:
    next_cabinet = _cabinet(
        "b",
        "Leader B",
        date(2020, 7, 1),
        date(2020, 6, 15),
    )
    mechanism, _ = classify_transition_mechanism(
        exit_date=date(2020, 7, 1),
        next_cabinet=next_cabinet,
        election_window_days=30,
    )
    assert mechanism == POST_ELECTION_TRANSITION
