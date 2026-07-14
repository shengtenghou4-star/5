# M3 competing-risk leader exits

M3 now separates two questions that were previously mixed together:

1. **How likely is the leader to leave office by each horizon?**
2. **Through which observable transition channel would that exit occur?**

The total exit curve remains the authority. Cause-specific models provide allocation signals, and the reconciliation layer guarantees that the mechanism probabilities add back to the total exit probability at every horizon.

## Current transition labels

ParlGov can support two auditable labels without inventing hidden political causes:

- `post_election_transition`: the next leader's cabinet begins within 120 days after its recorded election;
- `other_recorded_transition`: every other observed change of leader.

These are **transition channels**, not claims that an election, coalition revolt, protest, illness, coup, prosecution, or death caused the exit. The code deliberately refuses to infer those richer mechanisms from cabinet dates alone.

The 120-day window is configurable and recorded in the dataset manifest. An election dated after the leader transition is never used to label that transition as election-linked.

## Dataset structure

For each leader snapshot and horizon, the builder emits:

- one total-exit case, such as `government_leader_exit_90d`;
- one `post_election_transition` case;
- one `other_recorded_transition` case.

The cause-specific labels are mutually exclusive for a realized exit. A leader who remains in office through the full horizon is a negative case for both channels. If the leader exits through one channel, the other channel resolves false on the actual exit date.

All predictive features retain their original observation timestamps. Transition labels are outcomes only and never become features.

## Probability reconciliation

Independent binary models can produce impossible outputs: cause probabilities can exceed total exit risk, decrease at longer horizons, or fail to sum to the total. M3 fixes this in two stages:

1. apply the existing isotonic survival reconciliation to total exit probabilities;
2. allocate each interval's newly added exit mass across mechanisms using positive changes in the raw cause scores.

If all cause scores are flat, the current score mix is used. If every score is zero, the interval mass is split evenly rather than silently disappearing.

The resulting forecast guarantees:

- total exit probability never falls at a longer horizon;
- each mechanism's cumulative probability never falls;
- mechanism probabilities sum exactly to total exit probability;
- interval mechanism probabilities sum exactly to interval exit probability;
- survival probability plus total exit probability equals one.

## Commands

Build the combined total and mechanism dataset:

```bash
python -m fencha.m3_competing_cli build \
  --as-of 2023-06-30 \
  --horizons 30,90,180,365
```

Replay one historical target:

```bash
python -m fencha.m3_competing_cli forecast \
  --target-case-id 'parlgov:DEU:Merkel:2020-01-01:30d'
```

The forecast JSON states that the current labels are observable channels rather than asserted hidden causes.

## Next label expansion

The infrastructure accepts any mutually exclusive mechanism set. The next upgrade should add verified external event labels for channels such as:

- electoral defeat;
- coalition or party removal;
- voluntary resignation;
- mass protest pressure;
- military or unconstitutional removal;
- health or death;
- judicial removal.

Those labels require an event source with provenance and dates. They should not be generated from news keywords or cabinet timing alone.
