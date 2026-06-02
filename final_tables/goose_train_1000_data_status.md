# goose_train_1000 data preparation status

- images: 1000
- labels: 1000
- LISA traversable masks: 1000
- LISA pseudo labels thr50: 1000
- VL2G traversable geometry labels: 1000
- boundary width: 7
- uncertainty width: 15
- risk dir: None
- empty traversable: 0
- empty risk: 1000

Current purpose:
- Run 1000-scale LISA pseudo-only baseline
- Run 1000-scale VL2G boundary + smooth weak
- Evaluate both under strict and loose GT with threshold sweep
