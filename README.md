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
Highest Train: 0.9506±0.0003
Highest Valid: 0.7407±0.0013
  Final Train: 0.9504±0.0005
   Final Test: 0.7306±0.0016
```
