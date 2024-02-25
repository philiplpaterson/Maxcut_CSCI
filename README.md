
## Run algorithms

Format:
```
python greedy.py
python random_walk.py
python simulated_annealing.py
```
## Datasets

Take g14.txt as an example:

800 4694 

1 7 1 # node 1 connects node 7, and the weight is 1

1 10 1

1 12 1

1 13 1

The number of nodes is 800, and the number of edges is 4694.



## Store Results 

Results will be written to a file result.txt in the folder "result". The first column is the node, and the second column is the label of classified set. For example, 

1 2  # node 1 in set 2

2 1  # node 2 in set 1

3 2  # node 3 in set 2

4 1  # node 4 in set 1

5 2  # node 5 in set 2

The partial results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing).

## Performance
In the following experiments, we use GPU during training by default. The best-known results are labed by bold.


1) __Gset__

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Stanford university. 

| graph | #nodes| #edges | BLS | DSDP    | KHLWG   | RUN-CSP| PI-GNN| Gurobi (0.5 h)  | Gurobi (1 h)  |Gap           | Ours | Improvement | 
|--- |------|----  |---        |-----    |-----    |--------|-------|------           | ---           | ---           | ----| ----|
|G14 | 800  | 4694 | __3064__  |         | 2922    | 3061   | 2943  |3034             |3042           | 3.61\%        | __3064__ | +0\%|
|G15 | 800  | 4661 | __3050__  | 2938    |__3050__ | 2928   | 2990  | 3016            | 3033          |3.33\%         | __3050__ | +0\% | 
|G22 | 2000 | 19990|__13359__  | 12960   |__13359__| 13028  | 13181 |13062            |13129          | 28.94\%       | __13359__ |  +0\% | 
|G49 | 3000 | 6000 | __6000__  | __6000__|__6000__ |__6000__| 5918  |__6000__         |__6000__       |0              | __6000__|  +0\% | 
|G50 | 3000 | 6000 | __5880__  | __5880__|__5880__ |__5880__| 5820  |__5880__         |__5880__       |0              | __5880__|  +0\% | 
|G55 | 5000 | 12468| 10294     | 9960    | 10236   | 10116  | 10138 | 10103           | 10103         | 11.92\%       |__10298__ |  +0.04\% | 
|G70 | 10000| 9999 |9541       | 9456    | 9458    | -      | 9421  | 9489            | 9490          |2.26\%         |__9583__ | +0.44\% | 


2) __Syn__ 

We use the whole synthetic data. For graphs with n nodes, there are 5 datasets, and we run once for each dataset, and finally calcualte the average and standard deviation for the objective values. 
 
|Datasets |Gurobi (0.5 h)| Gap |Gurobi (1 h) | Gap |Gurobi (1.5 h) |Gap |Ours|Gap |
|-------|------|----| ---- |------|----| ---- |---- |--|
|syn_10   | 17.40 $\pm$ 0.80 (0.004s) | 0| 17.40 $\pm$ 0.80 (0.004s)| 0 | 17.40 $\pm$ 0.80 (0.004s)| 0| $\pm$  | |  
|syn_50   | 134.20 $\pm$ 2.04 (0.30s)  | 0| 134.20 $\pm$ 2.04 (0.30s)| 0  | 134.20 $\pm$ 2.04 (0.30s)| 0|  $\pm$   |  |  
|syn_100  |  337.20 $\pm$ 2.71 (289.99s) |0 | 337.20 $\pm$ 2.71 (289.99s)| 0 | 337.20 $\pm$ 2.71 (289.99s)| 0|  $\pm$ |   |  
|syn_300   | 1403.80 $\pm$ 7.44 (1800s)  | 9.58\%| 1404.00 $\pm$ 7.54 (3600s) | 9.18\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_500   |  2474.00 $\pm$ 13.89 (1800s)  | 13.93\%| 2475.40 $\pm$ 15.00 (3600s)| 13.86\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_700   |  2849.60 $\pm$ 14.08 (1800s)  | 13.55\%| 2852.2 $\pm$ 14.30 (3600s) | 13.26\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_900   |  3622.20 $\pm$ 11.84 (1800s)  | 14.32\% | 3624.00 $\pm$ 9.86 (3600s) | 13.88\%   | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_1000  |  4435.80 $\pm$ 18.14 (1800s)  | 15.95%| 4437.8 $\pm$ 16.85 (3600s) |  15.59\%   | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_3000  |  17111.00 $\pm$ 16.70 (1800s)  | 36.49\% | 17145.00 $\pm$ 33.60 (3600s) | 32.73\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_5000  |  30376.60 $\pm$ 243.14 (1800s)  | 54.83\% | 30500.80 $\pm$ 223.32 (3600s) | 52.17\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_7000  |  46978.60 $\pm$ 746.83 (1800s)  | 61.00\% | 47460.00 $\pm$ 473.76 (3600s) |  56.87\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_9000  |  57730.20 $\pm$ 502.51 (1800s)  | 60.30\% | 57730.20 $\pm$ 502.51 (3600s) | 60.00\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_10000 |  60768.40 $\pm$ 585.41 (1800s)  | 59.54\% | 60768.40 $\pm$ 585.41 (3600s) |  58.67\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  

