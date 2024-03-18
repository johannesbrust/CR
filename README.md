# CR
**Compact Representations**

Matlab and Python implementations from

"Useful Compact Representations for Data Fitting", J. Brust (2024),
[[Article](https://www.medrxiv.org/content/10.1101/2022.08.23.22279137v1 "Technical Report")]

Content:
  * MATLAB/
    * ALGS/
    * AUXILIARY/
    * EXAMPLES/
    * EXTERNAL/
  * PYTHON/  

Notes: The Matlab codes use external functions from `[1]` for tensor computations.
The Python code uses `[2]` for interfaces to neural networks.
    
## Example(s)

### Matlab
You can run a relatively small Matlab example from within MATLAB/EXAMPLES/

```
>> example_1

Compact Representation (ALG0)  

Line-search: wolfeG          
V-strategy:  s          
         n=  500          
--------------------------------
k    	 nf      	 fk         	 ||gk||         	 step        	 iexit       
0 	 1 	 2.50000e+02      	 3.16228e+01       	 1.000e+00     	 0
1 	 2 	 2.19777e+02      	 3.07453e+01       	 1.000e+00     	 1     
2 	 6 	 1.92234e+02      	 2.83866e+01       	 8.260e-03     	 1     
3 	 9 	 1.62892e+02      	 8.67488e+01       	 1.101e-01     	 1     
4 	 10 	 1.27783e+02      	 7.62049e+01       	 1.000e+00     	 1     
5 	 11 	 1.05840e+02      	 9.22853e+01       	 1.000e+00     	 1     
6 	 13 	 6.36547e+01      	 4.52269e+01       	 5.470e-01     	 1     
7 	 15 	 6.10026e+01      	 1.12860e+01       	 1.175e-01     	 1     
8 	 18 	 4.03289e+01      	 7.76874e+01       	 1.550e-01     	 1     
9 	 20 	 3.45282e+01      	 7.33576e+00       	 1.498e-01     	 1     
10 	 24 	 2.88600e+01      	 6.87921e+00       	 4.095e-03     	 1     
11 	 26 	 2.84872e+01      	 1.89809e+01       	 1.085e-01     	 1     
12 	 31 	 1.79508e+01      	 1.59878e+01       	 2.296e-02     	 1     
13 	 33 	 1.77297e+01      	 4.95834e+00       	 2.767e-02     	 1     
14 	 37 	 1.42409e+01      	 4.98738e+00       	 1.699e-03     	 1     
15 	 39 	 1.40577e+01      	 1.89934e+01       	 5.026e-02     	 1     
16 	 43 	 1.10959e+01      	 2.06549e+01       	 2.759e-03     	 1     
17 	 45 	 1.09185e+01      	 3.60611e+01       	 8.882e-02     	 1     
18 	 49 	 7.98050e+00      	 3.13865e+01       	 1.666e-03     	 1     
19 	 50 	 3.54884e+00      	 3.69810e+01       	 1.000e+00     	 1     
k    	 nf      	 fk         	 ||gk||         	 step        	 iexit       
20 	 52 	 2.55290e+00      	 5.51104e+00       	 4.003e-05     	 1     
21 	 54 	 7.43328e-01      	 2.52369e+01       	 4.262e-01     	 1     
22 	 56 	 4.04685e-01      	 5.87751e-01       	 2.686e-02     	 1     
23 	 60 	 2.80015e-01      	 5.09780e-01       	 1.018e-02     	 1     
24 	 62 	 2.60442e-01      	 6.04220e+00       	 8.381e-02     	 1     
25 	 66 	 1.26708e-01      	 4.75861e+00       	 7.317e-03     	 1     
26 	 68 	 1.23042e-01      	 1.93417e+00       	 4.829e-03     	 1     
27 	 71 	 4.54252e-03      	 2.96931e+00       	 1.064e-01     	 1     
28 	 73 	 2.63784e-04      	 1.66347e-02       	 1.443e-04     	 1     
29 	 75 	 4.69816e-07      	 3.06742e-02       	 3.013e-01     	 1     
30 	 77 	 1.13207e-10      	 2.02066e-05       	 1.882e-03     	 1 
```

A larger example is given by ``>> example_2``

### Python
Likewise, in the Julia REPL, navigate to the JULIA/EXAMPLES folder. 
Run the following from within the folder (tested with Julia 1.6.5):

```
julia> include("../AUXILIARY/setup.jl"); # setup the dependencies

```

Run examples equivalent to MATLAB:

```
julia> include("example_1.jl");

----------- COMP Algorithm ----------- 

Problem Size
Tests:                 20
Samples:               64
Input Expct. Positive: 2 

OUTPUTS:################
Time (search):         5.6223e-5
Positive items:        2
########################
Identified Indices:    [2, 13]

```

## Cite
If this work is useful to you, you can cite this work as (bibtex)

```
@article{brust24,
  title       = {Useful Compact Representations for Data-Fitting},
  author      = {Brust, Johannes J},
  institution = {School of Mathematical and Statistical Sciences, Arizona State University, Tempe, AZ},
  type        = {Technical Report},
  year        = {2024},
  url         = {https://doi.org/10.48550/arXiv.2209.12057}
}
```

## Reference codes
[1]  Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.6, www.tensortoolbox.org, September 28, 2023. 

[2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035), https://pytorch.org/, March 18, 2024.
