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
You can run a relatively small Matlab example from within MATLAB/

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

A larger experiment is given in MATLAB/EXPERIMENTS/ML_REGRESSION/ ``>> mnist_train``

```
>> mnist_train

#########################################################
#
# MNIST Dataset for digit classification 
# 
# N    = 60000 	 (data size) 
# nx   = 784 	 (img pixels) 
# Btch = 256 	 (batch size) 
# d    = 7840 	 (num. vars) 
#
# Alg  = compact 	 (algorithm) 
# 
#########################################################
Epoch   	 Iter    	 Loss    	 Accr    	 norm(w) 	 norm(g) 
1       	 235     	 0.418   	 0.904   	  7.3   	  0.488   	 
2       	 470     	 0.377   	 0.912   	  8.7   	  0.432   	 
3       	 705     	 0.357   	 0.914   	  9.61   	  0.411   	 
4       	 940     	 0.343   	 0.915   	  10.3   	  0.399   	 
5       	 1175     	 0.334   	 0.917   	  10.9   	  0.389   	 
6       	 1410     	 0.327   	 0.917   	  11.5   	  0.382   	 
7       	 1645     	 0.32   	 0.917   	  11.9   	  0.377   	 
8       	 1880     	 0.315   	 0.919   	  12.3   	  0.373   	 
9       	 2115     	 0.31   	 0.919   	  12.7   	  0.368   	 
10       	 2350     	 0.306   	 0.919   	  13.1   	  0.364   	 
11       	 2585     	 0.302   	 0.92   	  13.4   	  0.36   	 
12       	 2820     	 0.297   	 0.92   	  13.8   	  0.355   	 
13       	 3055     	 0.294   	 0.921   	  14.1   	  0.351   	 
14       	 3290     	 0.29   	 0.921   	  14.4   	  0.347   	 
15       	 3525     	 0.286   	 0.921   	  14.7   	  0.344   	 
16       	 3760     	 0.283   	 0.921   	  15   	  0.341   	 
17       	 3995     	 0.279   	 0.921   	  15.2   	  0.338   	 
18       	 4230     	 0.276   	 0.921   	  15.5   	  0.335   	 
19       	 4465     	 0.273   	 0.921   	  15.7   	  0.332   	 
20       	 4700     	 0.27   	 0.921   	  16   	  0.33  
```

In general, you can run the Matlab experiments from within MATLAB/EXPERIMENTS/ 

### Python
Likewise, in Python you can run the compact solver (tested with Python 3.11.8)

Note: In order to run the method for a Fashion MNIST classification,
both `PyTorch` and its library `torchvision` have to be installed.
(For managing the packages one can use Anaconda or related package managers)

From within the PYTHON folder

```
 python3 fashionMnistCOMP.py 
```

yields the output

```
Epoch 1
-------------------------------
loss: 2.303020  [   64/60000]
loss: 0.712614  [ 6464/60000]
loss: 0.475621  [12864/60000]
loss: 0.649297  [19264/60000]
loss: 0.521813  [25664/60000]
loss: 0.452051  [32064/60000]
loss: 0.434324  [38464/60000]
loss: 0.551466  [44864/60000]
loss: 0.585781  [51264/60000]
loss: 0.669467  [57664/60000]
Test Error: 
 Accuracy: 82.8%, Avg loss: 0.464052 

Epoch 2
-------------------------------
loss: 0.359722  [   64/60000]
loss: 0.387545  [ 6464/60000]
loss: 0.299054  [12864/60000]
loss: 0.395482  [19264/60000]
loss: 0.369756  [25664/60000]
loss: 0.435838  [32064/60000]
loss: 0.333493  [38464/60000]
loss: 0.477194  [44864/60000]
loss: 0.474102  [51264/60000]
loss: 0.578183  [57664/60000]
Test Error: 
 Accuracy: 85.0%, Avg loss: 0.413415
....
....
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
