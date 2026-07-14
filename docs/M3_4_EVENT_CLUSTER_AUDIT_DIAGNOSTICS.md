# M3.4 Event-Cluster Audit Diagnostics

Source commit: `05e04a461af37e74ba947c7259425c181e5d8fe7`

- Audit status: failure
- Forecast cases built: 317232
- Leader spells: 212
- Recorded transition counts: {'other_recorded_transition': 61, 'post_election_transition': 114}

## Audit log tail

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/runner/work/5/5/src/fencha/m3_4_event_audit_cli.py", line 143, in <module>
    main()
  File "/home/runner/work/5/5/src/fencha/m3_4_event_audit_cli.py", line 103, in main
    report = temporal_event_path_audit(
   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/5/5/src/fencha/event_path_audit.py", line 526, in temporal_event_path_audit
    raise ValueError("mechanism histories contain different exit events")
ValueError: mechanism histories contain different exit events
```
