# stepchange/ — the step-change mechanism experiment

## Files

| File | What it is |
|---|---|
| `stepchange_results.json` | B=280, const_weight@price 50 (the deployed operating point; Defect #7 re-run, 2026-10-06) |
| `stepchange_results_B320.json` | B=320, const_weight@price 50 (the second budget) |
| `stepchange_results_price1_OLD.json` | B=280, const_weight@price 1.0 (the original run, superseded) |
| `stepchange_results_B320_price1_OLD.json` | B=320, const_weight@price 1.0 (the original run, superseded) |
| `map.yaml` | The hand-authored 30×6 m step-change map |
| `certificate.json` | The map's fitted certificate (a_q = 53.54 J, coverage 0.950) |
| `stepchange_speeds.pdf` + provenance | The speed-profile figure (Fig. 9) |

## History

1. **2026-09-12:** First run (pre-stall-retry code), quarantined at
   `../stepchange_prefix_20260912/`.
2. **2026-09-15:** Re-run under final code (script 22) at const_weight
   price 1.0 (the evaluator default). Reproduced the 09-12 results
   seed-for-seed; mechanism confirmed code-version-stable.
3. **2026-10-06:** Defect #7 consistency fix — the const_weight arm was
   re-run at the sweep-selected price 50 (matching the campaign grid).
   Both budgets re-run (script 22 with `--const-price 50`, overnight).
   **Mechanism confirmed unchanged:** the priced arm at price 50 still
   cruises through the step boundary at full speed and turns 3–4 m too
   late. The reach arm still decelerates and turns at the certified
   boundary. Per-seed numbers shifted slightly; the paper's §VII-E
   quotes the price-50 results.

## Driver

```bash
python scripts/22_reach_stepchange.py --run-id <RUN_ID> --const-price 50 --budget 280
python scripts/22_reach_stepchange.py --run-id <RUN_ID> --const-price 50 --budget 320
```
