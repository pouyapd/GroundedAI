| Metric | extractive-demo-v1 · answer-v2 (e928c4235d03) | Qwen/Qwen2.5-0.5B-Instruct · answer-v1 | Qwen/Qwen2.5-0.5B-Instruct · answer-v2 (e928c4235d03) | qwen2.5:1.5b · answer-v1 (93f4b2d66ffa) | qwen2.5:1.5b · answer-v2 (e928c4235d03) |
|---|---|---|---|---|---|
| Questions | 56 | 56 | 56 | 56 | 56 |
| Answered | 38 | 8 | 25 | 37 | 49 |
| Abstained | 18 | 44 | 23 | 19 | 7 |
| Invalid output (markers only or empty) | 0 | 4 | 8 | 0 | 0 |
| Errors | 0 | 0 | 0 | 0 | 0 |
| Timeouts | 0 | 0 | 0 | 0 | 0 |
| Correct behaviour (scored questions) | 0.660 | 0.340 | 0.528 | 0.830 | 0.906 |
| Answer rate, answerable questions | 0.698 | 0.186 | 0.488 | 0.837 | 1.000 |
| Incorrect refusal rate, answerable questions | 0.302 | 0.721 | 0.372 | 0.163 | 0.000 |
| Correct abstention rate (should-refuse questions) | 0.500 | 1.000 | 0.700 | 0.900 | 0.700 |
| Abstention precision | 0.278 | 0.227 | 0.304 | 0.474 | 1.000 |
| Conflict questions: both sides cited | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Conflict questions: disagreement stated | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Key-fact recall | 0.558 | 0.163 | 0.302 | 0.767 | 0.942 |
| Citation-format compliance (answers) | 1.000 | 0.125 | 0.080 | 0.297 | 0.265 |
| Citation validity (answers) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Evidence support (cited sentences) | 1.000 | – | 0.545 | 0.714 | 0.722 |
| Evidence coverage | 0.663 | 0.023 | 0.186 | 0.221 | 0.209 |
| Answers with malformed citations | 0.000 | 0.750 | 0.880 | 0.054 | 0.245 |
| Invalid output rate | 0.000 | 0.071 | 0.143 | 0.000 | 0.000 |
| Injection resistance | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 |
| Latency p50 / p95 | 0.0 s / 0.0 s | 32.3 s / 52.3 s | 38.0 s / 47.0 s | 10.9 s / 20.1 s | 10.0 s / 16.4 s |
| Answers re-scored as refusals (current parser) | 0 | 2 | 0 | 0 | 0 |

Paired differences, Qwen/Qwen2.5-0.5B-Instruct · answer-v1 minus extractive-demo-v1 · answer-v2 (e928c4235d03):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 53 | -0.321 | [-0.509, -0.113] | yes |
| Answered (answerable questions) | 43 | -0.512 | [-0.698, -0.326] | yes |
| Refused (answerable questions) | 43 | +0.419 | [+0.209, +0.628] | yes |
| Key-fact recall | 43 | -0.395 | [-0.558, -0.232] | yes |

Paired differences, Qwen/Qwen2.5-0.5B-Instruct · answer-v2 (e928c4235d03) minus extractive-demo-v1 · answer-v2 (e928c4235d03):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 53 | -0.132 | [-0.302, +0.038] | no |
| Answered (answerable questions) | 43 | -0.209 | [-0.395, -0.023] | yes |
| Refused (answerable questions) | 43 | +0.070 | [-0.116, +0.256] | no |
| Key-fact recall | 43 | -0.256 | [-0.430, -0.081] | yes |

Paired differences, qwen2.5:1.5b · answer-v1 (93f4b2d66ffa) minus extractive-demo-v1 · answer-v2 (e928c4235d03):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 53 | +0.170 | [+0.000, +0.340] | no |
| Answered (answerable questions) | 43 | +0.140 | [-0.047, +0.326] | no |
| Refused (answerable questions) | 43 | -0.140 | [-0.326, +0.047] | no |
| Key-fact recall | 43 | +0.209 | [+0.035, +0.384] | yes |

Paired differences, qwen2.5:1.5b · answer-v2 (e928c4235d03) minus extractive-demo-v1 · answer-v2 (e928c4235d03):

| Measure | n | Difference | 95% interval | Distinguishable |
|---|---|---|---|---|
| Correct behaviour | 53 | +0.245 | [+0.113, +0.377] | yes |
| Answered (answerable questions) | 43 | +0.302 | [+0.163, +0.442] | yes |
| Refused (answerable questions) | 43 | -0.302 | [-0.442, -0.163] | yes |
| Key-fact recall | 43 | +0.384 | [+0.244, +0.523] | yes |
