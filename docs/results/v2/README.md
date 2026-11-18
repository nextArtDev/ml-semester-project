# Version 2 benchmark artifacts (superseded)

These are the unmodified artifacts from the previous 42-run benchmark (three
balanced rounds, 1,000 stratified queries, three repetitions). They are kept
so the effect of evaluating the full 3,000-query held-out pool can be compared
without rewriting history.

They used:

```
--dataset-id 116 --sizes 10000,50000 --ef-search 10,25,50,100,200,400
--k 5 --metric l2 --max-queries 1000 --repetitions 3 --rounds 3
--order-seed 0 --query-sampling stratified --query-sample-seed 0 --seed 0
```

The current artifacts in `..` use `--max-queries 0` (all 3,000 held-out rows),
explicit `--run-id`, `benchmark_orders.json`, and the revised timing fields.
Use the current files for new claims.
