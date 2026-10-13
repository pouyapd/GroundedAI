| Metric | Qwen/Qwen2.5-0.5B-Instruct · answer-v1 | Qwen/Qwen2.5-0.5B-Instruct · answer-v2 | Qwen/Qwen2.5-0.5B-Instruct · answer-v2 (ed0392f23f81) |
|---|---|---|---|
| Questions | 25 | 25 | 25 |
| Answered | 3 | 17 | 12 |
| Abstained | 22 | 8 | 13 |
| Invalid output (markers only or empty) | 0 | 0 | 0 |
| Errors | 0 | 0 | 0 |
| Timeouts | 0 | 0 | 0 |
| Correct behaviour (scored questions) | 0.280 | 0.680 | 0.520 |
| Answer rate, answerable questions | 0.105 | 0.789 | 0.526 |
| Incorrect refusal rate, answerable questions | 0.895 | 0.210 | 0.474 |
| Correct abstention rate (should-refuse questions) | 0.833 | 0.667 | 0.667 |
| Abstention precision | 0.227 | 0.500 | 0.308 |
| Conflict questions: both sides cited | 0.000 | 0.000 | 0.000 |
| Conflict questions: disagreement stated | 0.000 | 0.000 | 0.000 |
| Key-fact recall | 0.105 | 0.368 | 0.184 |
| Citation-format compliance (answers) | 0.333 | 0.235 | 0.000 |
| Citation validity (answers) | 1.000 | 1.000 | 1.000 |
| Evidence support (cited sentences) | 1.000 | 0.250 | – |
| Evidence coverage | 0.053 | 0.105 | 0.000 |
| Answers with malformed citations | 0.667 | 0.235 | 0.750 |
| Invalid output rate | 0.000 | 0.000 | 0.000 |
| Injection resistance | 1.000 | 1.000 | 1.000 |
| Latency p50 / p95 | 38.7 s / 51.3 s | 39.7 s / 47.1 s | 36.7 s / 51.2 s |
| Answers re-scored as refusals (current parser) | 4 | 0 | 0 |

Paired differences, Qwen/Qwen2.5-0.5B-Instruct · answer-v2 minus Qwen/Qwen2.5-0.5B-Instruct · answer-v1:

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 25 | +0.400 | [+0.160, +0.640] | yes |
| Answered (answerable questions) | 19 | +0.684 | [+0.474, +0.895] | yes |
| Refused (answerable questions) | 19 | -0.684 | [-0.895, -0.474] | yes |
| Key-fact recall | 19 | +0.263 | [+0.053, +0.474] | yes |

Paired differences, Qwen/Qwen2.5-0.5B-Instruct · answer-v2 (ed0392f23f81) minus Qwen/Qwen2.5-0.5B-Instruct · answer-v1:

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 25 | +0.240 | [+0.000, +0.480] | no |
| Answered (answerable questions) | 19 | +0.421 | [+0.211, +0.632] | yes |
| Refused (answerable questions) | 19 | -0.421 | [-0.632, -0.211] | yes |
| Key-fact recall | 19 | +0.079 | [-0.053, +0.237] | no |
