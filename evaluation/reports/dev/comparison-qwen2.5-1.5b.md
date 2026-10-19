| Metric | qwen2.5:1.5b · answer-v1 (93f4b2d66ffa) | qwen2.5:1.5b · answer-v2 (ed0392f23f81) | qwen2.5:1.5b · answer-v2 (e928c4235d03) | qwen2.5:1.5b · answer-v2 (94f3994f327f) |
|---|---|---|---|---|
| Questions | 25 | 25 | 25 | 25 |
| Answered | 13 | 21 | 19 | 17 |
| Abstained | 12 | 4 | 6 | 8 |
| Invalid output (markers only or empty) | 0 | 0 | 0 | 0 |
| Errors | 0 | 0 | 0 | 0 |
| Timeouts | 0 | 0 | 0 | 0 |
| Correct behaviour (scored questions) | 0.720 | 0.840 | 0.880 | 0.760 |
| Answer rate, answerable questions | 0.684 | 1.000 | 0.947 | 0.842 |
| Incorrect refusal rate, answerable questions | 0.316 | 0.000 | 0.053 | 0.158 |
| Correct abstention rate (should-refuse questions) | 1.000 | 0.667 | 0.833 | 0.833 |
| Abstention precision | 0.500 | 1.000 | 0.833 | 0.625 |
| Conflict questions: both sides cited | 0.500 | 0.000 | 0.500 | 0.000 |
| Conflict questions: disagreement stated | 0.000 | 0.000 | 0.000 | 0.000 |
| Key-fact recall | 0.579 | 0.763 | 0.895 | 0.737 |
| Citation-format compliance (answers) | 0.538 | 0.286 | 0.421 | 0.294 |
| Citation validity (answers) | 1.000 | 1.000 | 1.000 | 1.000 |
| Evidence support (cited sentences) | 0.714 | 0.500 | 0.667 | 0.571 |
| Evidence coverage | 0.342 | 0.158 | 0.342 | 0.289 |
| Answers with malformed citations | 0.077 | 0.619 | 0.316 | 0.353 |
| Invalid output rate | 0.000 | 0.000 | 0.000 | 0.000 |
| Injection resistance | 1.000 | 0.500 | 0.500 | 1.000 |
| Latency p50 / p95 | 11.0 s / 19.7 s | 10.2 s / 13.2 s | 10.8 s / 18.2 s | 10.8 s / 13.4 s |
| Answers re-scored as refusals (current parser) | 0 | 2 | 1 | 0 |

Paired differences, qwen2.5:1.5b · answer-v2 (ed0392f23f81) minus qwen2.5:1.5b · answer-v1 (93f4b2d66ffa):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 25 | +0.120 | [-0.120, +0.360] | no |
| Answered (answerable questions) | 19 | +0.316 | [+0.105, +0.526] | yes |
| Refused (answerable questions) | 19 | -0.316 | [-0.526, -0.105] | yes |
| Key-fact recall | 19 | +0.184 | [-0.079, +0.447] | no |

Paired differences, qwen2.5:1.5b · answer-v2 (e928c4235d03) minus qwen2.5:1.5b · answer-v1 (93f4b2d66ffa):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 25 | +0.160 | [+0.000, +0.360] | no |
| Answered (answerable questions) | 19 | +0.263 | [+0.105, +0.474] | yes |
| Refused (answerable questions) | 19 | -0.263 | [-0.474, -0.105] | yes |
| Key-fact recall | 19 | +0.316 | [+0.105, +0.526] | yes |

Paired differences, qwen2.5:1.5b · answer-v2 (94f3994f327f) minus qwen2.5:1.5b · answer-v1 (93f4b2d66ffa):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 25 | +0.040 | [-0.160, +0.240] | no |
| Answered (answerable questions) | 19 | +0.158 | [-0.053, +0.368] | no |
| Refused (answerable questions) | 19 | -0.158 | [-0.368, +0.053] | no |
| Key-fact recall | 19 | +0.158 | [-0.105, +0.395] | no |
