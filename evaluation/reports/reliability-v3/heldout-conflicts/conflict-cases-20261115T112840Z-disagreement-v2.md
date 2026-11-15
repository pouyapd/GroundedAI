# Disagreement check on the held-out cases (groundedai-conflicts-v1)

Generated 2026-11-15T11:28:40+00:00 · code `73145f40ad5f (uncommitted changes)` · cases `256eaad3e44a`

| Check | Disagreements found (of 16) | Kind also right | False alarms (of 14) |
|---|---|---|---|
| disagreement-v2 | 13 | 11 | 0 |
| figures-v1 (published 2026-10-13) | 6 | – | 3 |

| Class | Cases | disagreement-v2 | figures-v1 |
|---|---|---|---|
| conflict | 10 | found 8, kind right 8 | found 2 |
| temporal | 6 | found 5, kind right 3 | found 4 |
| context | 6 | false alarms 0 | false alarms 3 |
| agreement | 4 | false alarms 0 | false alarms 0 |
| incomplete | 4 | false alarms 0 | false alarms 0 |

## Cases

| Case | Class | disagreement-v2 | figures-v1 |
|---|---|---|---|
| c01 | conflict | conflict S1/S2 | – |
| c02 | conflict | conflict S1/S2 | – |
| c03 | conflict | conflict S1/S2 | – |
| c04 | conflict | conflict S1/S2 | – |
| c05 | conflict | conflict S1/S2 | – |
| c06 | conflict | conflict S1/S2 | S1/S2 |
| c07 | conflict | conflict S1/S2 | – |
| c08 | conflict | – | S1/S2 |
| c09 | conflict | – | – |
| c10 | conflict | conflict S1/S2 | – |
| t01 | temporal | different_dates S1/S2 | S1/S2 |
| t02 | temporal | different_dates S1/S2 | – |
| t03 | temporal | conflict S1/S2 | S1/S2 |
| t04 | temporal | conflict S1/S2 | S1/S2 |
| t05 | temporal | different_dates S1/S2 | S1/S2 |
| t06 | temporal | – | – |
| x01 | context | – | – |
| x02 | context | – | S1/S2 |
| x03 | context | – | – |
| x04 | context | – | S1/S2 |
| x05 | context | – | – |
| x06 | context | – | S1/S2 |
| a01 | agreement | – | – |
| a02 | agreement | – | – |
| a03 | agreement | – | – |
| a04 | agreement | – | – |
| i01 | incomplete | – | – |
| i02 | incomplete | – | – |
| i03 | incomplete | – | – |
| i04 | incomplete | – | – |

## Limitations

- 30 cases written by the same author as the check; small counts, no intervals.
- The cases test the check on its own, with the passages given; retrieval is not involved.
