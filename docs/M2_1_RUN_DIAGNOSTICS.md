# M2.1 Data Preparation Diagnostics

## ParlGov build tail

```text
cabinets=1618 cases=26436 positives=817 countries=37 output=data/processed/parlgov_leader_exit_180d.jsonl manifest=data/processed/parlgov_leader_exit_180d.manifest.json
```

## GDELT build tail

```text
GDELT downloads: requested=157 ok=0 cache_hits=0 missing=157 missing_rate=100.0% errors={'tls_error': 157}
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/runner/work/5/5/src/fencha/m2_cli.py", line 248, in <module>
    main()
  File "/home/runner/work/5/5/src/fencha/m2_cli.py", line 242, in main
    print(_build(args))
          ^^^^^^^^^^^^
  File "/home/runner/work/5/5/src/fencha/m2_cli.py", line 174, in _build
    raise RuntimeError(str(manifest["failure_reason"]))
RuntimeError: GDELT missing rate 100.0% exceeds limit 15.0%
```

## GDELT manifest summary

```json
{
  "status": "failed_missing_rate",
  "requested_files": 157,
  "downloaded_files": 0,
  "cache_hits": 0,
  "missing_files": 157,
  "missing_rate": 1.0,
  "error_summary": {
    "tls_error": 157
  }
}
```
