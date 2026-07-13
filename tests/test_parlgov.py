from datetime import date
from io import StringIO

from fencha.datasets.parlgov import (
    PARLGOV_CSV_URL,
    build_leader_exit_cases,
    infer_leader_name,
    read_cabinets,
)

CSV = """country_name_short,country_name,cabinet_id,cabinet_name,start_date,election_date,end_date,party_id,seats,seats_total,cabinet,caretaker,cabinet_type
AAA,Alpha,1,Smith I,2018-01-15,2017-12-01,2019-05-01,p1,55,100,1,0,majority
AAA,Alpha,1,Smith I,2018-01-15,2017-12-01,2019-05-01,p2,10,100,1,0,majority
AAA,Alpha,1,Smith I,2018-01-15,2017-12-01,2019-05-01,p9,35,100,0,0,majority
AAA,Alpha,2,Smith II,2019-05-01,2019-04-01,2020-09-10,p1,48,100,1,0,minority
AAA,Alpha,3,Jones I,2020-09-10,2020-08-01,,p3,60,100,1,0,majority
BBB,Beta,4,Lee caretaker,2020-01-01,2019-12-01,2020-04-01,p4,20,50,1,1,caretaker
BBB,Beta,5,Patel I,2020-04-01,2020-03-01,,p5,30,50,1,0,majority
"""


def test_default_source_is_official_view_cabinet_csv() -> None:
    assert PARLGOV_CSV_URL.endswith("/view_cabinet.csv")


def test_infer_leader_name_strips_cabinet_suffixes() -> None:
    assert infer_leader_name("Smith III") == "Smith"
    assert infer_leader_name("Lee caretaker") == "Lee"


def test_parser_aggregates_only_government_party_rows() -> None:
    cabinets = read_cabinets(StringIO(CSV))
    assert len(cabinets) == 5
    smith = cabinets[0]
    assert smith.leader_name == "Smith"
    assert smith.coalition_size == 2
    assert smith.government_seats == 65
    assert smith.parliament_seats == 100


def test_builder_tracks_same_leader_across_cabinets_and_avoids_censoring() -> None:
    cabinets = read_cabinets(StringIO(CSV))
    cases = build_leader_exit_cases(
        cabinets,
        as_of=date(2022, 1, 1),
        earliest_cutoff=date(2018, 1, 1),
    )
    smith_cases = [case for case in cases if ":Smith:" in case.case_id]
    assert smith_cases
    assert any(case.cutoff_at.date() > date(2019, 5, 1) for case in smith_cases)
    assert any(case.outcome for case in smith_cases)

    current_cases = [case for case in cases if ":Jones:" in case.case_id]
    assert current_cases
    assert all(not case.outcome for case in current_cases)
    assert all(case.resolved_at.date() <= date(2022, 1, 1) for case in current_cases)


def test_cases_are_time_safe() -> None:
    cases = build_leader_exit_cases(
        read_cabinets(StringIO(CSV)),
        as_of=date(2022, 1, 1),
    )
    assert len(cases) > 30
    assert all(
        feature.observed_at <= case.cutoff_at
        for case in cases
        for feature in case.features.values()
    )
