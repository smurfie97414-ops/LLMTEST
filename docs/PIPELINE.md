# Cycle pipeline

The new cycle layer turns the plan into an executable evaluation loop.

```text
reference model
trial model
  -> dynamic verifier
  -> regression list
  -> ledgers
  -> regression analysis
  -> budgeted actions
  -> extra generated checks
  -> markdown report
```

Run:

```bash
python tools/run_cycle_report.py
```

Main files:

- `cortex3_phases.py`: phase registry.
- `cortex3_ledgers.py`: bit, skill, causal and uncertainty ledgers.
- `cortex3_analysis.py`: failure cause hints.
- `cortex3_cycle.py`: end-to-end cycle report.
- `cortex3_selection.py`: offline trial selection.
