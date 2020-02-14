a reading note from fast.ai lesson 2
# Using a GPU on Google Cloud Platform
AI platform $/rightarrow$ new instance $/rightarrow$ pytorch 1.3 with a GPU
```python
import torch
t_cpu = torch.rand(500,500,500)
%timeit t_cpu @ t_cpu

t_gpu = torch.rand(500,500,500).cuda()
%timeit t_gpu @ t_gpu
```
#  Jupyter Notebook
1. import the packages
2. download the data from google chrome use some Javascript
```Java
urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```
3. ImageDataBunch help to build the training set and validation set
4. train the classification model `lear.fit_one_cylce`
5. plot and get the best learning rate
6. clean your data, delete the top loss images
7. put it into product turn it into web applications


**Problems:**  
* Learning Rate (3e-3 is usually good)
  * Low learning rate: very very slow study, and the training loss will be higher than valdation loss, it means that it haven't fit enough.  
* Number of Epochs
  * Too less epochs: simmilar as low learning rate.
  * Too many epochs: over fitting, it learn to recognize particular things but not the things in ganeral. (hard to overfitting with DL)
    it performs that good for a while and getting worse again.
  **Not True: training loss < validation loss $/rightarrow$ over fitting**  
  
# SGD
a web: http://matrixmultiplication.xyz
most time we need less data than we think, unbalanced data also works
