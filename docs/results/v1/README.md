# Version 1 benchmark artifacts (superseded)

These files are the **original, unmodified** results produced before the
benchmark-quality audit. They are kept so that the old numbers can be compared
with the new run and so that no historical result is silently rewritten.

They were generated with:

```
--sizes 10000,50000 --ef-search 10,25,50,100,200,400 --k 5 --metric l2
--max-queries 1000 --repetitions 3 --seed 0
```

Differences from the current run in `..`:

| Aspect | v1 | current |
| --- | --- | --- |
| Configuration order | fixed: exact, then ascending `ef_search` | seeded balanced rounds; every configuration runs early and late |
| Query selection | first 1,000 test ids | seeded stratified sample of 1,000 of 3,000 held-out rows |
| Repetitions per configuration | 3 | 3 |
| Rounds | 1 | 3 |
| Timing | `api_overhead_s` = runtime minus query time | explicit non-overlapping phase decomposition |
| Exact ordering | `ORDER BY distance` | `ORDER BY distance, id` |

The v1 numbers are **not** directly comparable with the current numbers
because the execution order and the query sample changed. Use this directory
only for historical comparison; quote the current artifacts for new claims.
