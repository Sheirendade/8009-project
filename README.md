# 8009-project
This is an improvement of baesline on the ogbn-arxiv dataset.

## Dependencies
Tested on windows

+ python == 3.9.11
+ pytorch == 1.11.1
+ pytorch_geometric == 2.0.4
+ ogb == 1.3.3

## Dataset
ogbn-arxiv

## Running the experiment
The model is 6 layers, and runs 500 epochs, run:

```bash
python sage_res_6layers_Smooth.py
```

## Result:

```bash
All runs:
Highest Train: 96.79 ± 0.08
Highest Valid: 73.98 ± 0.10
  Final Train: 95.04 ± 0.05
   Final Test: 72.90 ± 0.27
```
