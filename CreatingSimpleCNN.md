```python
def conv(ni, nf):
  return nn.Conv2d(ni,nf,kernel_size=3, stride=2, pedding=1)
  # stride: skipping over pixels. The image here is 28*28
model = nn.Sequential(
  conv(1, 8), # transform from 28*28 into 14*14
  nn.BatchNorm2d(8),
  nn.ReLU(),
  conv(8, 16), # 7
  nn.BatchNorm2d(16),
  nn.ReLU(),
  conv(16, 32), # 4
  nn.BatchNorm2d(32),
  nn.ReLU(),
  conv(32, 16), # 2 
  nn.BatchNorm2d(16),
  nn.ReLU(),
   conv(16, 10), # 1
  nn.BatchNorm2d(10),
  nn.ReLU(),
  Flatten()
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit_one_cycle(10,max_lr=0.1)
```
