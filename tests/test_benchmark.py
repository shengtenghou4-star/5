from datetime import date, datetime, timezone
from io import StringIO

from fencha.benchmark import temporal_holdout_benchmark
from fencha.caseio import read_jsonl
from fencha.datasets.parlgov import (
    build_leader_exit_cases,
    read_cabinets,
    write_jsonl,
)
from fencha.engine import AnalogForecaster

CSV = """country_name_short,country_name,cabinet_id,cabinet_name,start_date,election_date,end_date,party_id,seats,seats_total,cabinet_party,caretaker,cabinet_type
AAA,Alpha,1,Smith I,2010-01-15,2009-12-01,2012-05-01,p1,55,100,1,0,majority
AAA,Alpha,1,Smith I,2010-01-15,2009-12-01,2012-05-01,p2,10,100,1,0,majority
AAA,Alpha,2,Smith II,2012-05-01,2012-04-01,2014-09-10,p1,48,100,1,0,minority
AAA,Alpha,3,Jones I,2014-09-10,2014-08-01,2017-06-01,p3,60,100,1,0,majority
AAA,Alpha,4,Garcia I,2017-06-01,2017-05-01,,p4,52,100,1,0,majority
BBB,Beta,5,Lee caretaker,2011-01-01,2010-12-01,2011-04-01,p5,20,50,1,1,caretaker
BBB,Beta,6,Patel I,2011-04-01,2011-03-01,2015-03-01,p6,30,50,1,0,majority
BBB,Beta,7,Kim I,2015-03-01,2015-02-01,2018-01-01,p7,24,50,1,0,minority
BBB,Beta,8,Nova I,2018-01-01,2017-12-01,,p8,28,50,1,0,majority
"""


def test_jsonl_roundtrip_and_temporal_benchmark(tmp_path) -> None:
    cases = build_leader_exit_cases(
        read_cabinets(StringIO(CSV)),
        as_of=date(2021, 1, 1),
    )
    path = tmp_path / "cases.jsonl"
    write_jsonl(cases, path)
    loaded = read_jsonl(path)
    assert loaded == cases

    report = temporal_holdout_benchmark(
        loaded,
        AnalogForecaster(top_k=20),
        holdout_start=datetime(2016, 1, 1, tzinfo=timezone.utc),
        minimum_training_cases=20,
        target_stride=2,
        max_history=200,
    )
    assert report.baseline.predictions == report.analog.predictions
    assert report.baseline.predictions > 0
    assert 0 <= report.baseline.brier_score <= 1
    assert 0 <= report.analog.brier_score <= 1
    assert report.analog_brier_skill < 1
