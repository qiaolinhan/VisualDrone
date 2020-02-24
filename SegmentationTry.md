learning stuffs from http://fast.ai
# lesson 3 Multilabel Classificaton and Segmentation
On **Coursera**, course of Machine learning from Andrew looks good.  
variant analysis and false positives-- `F1-SCORE` 
Often, we can get our data from kaggle.
* kaggle for dataset

turn the data into DataBunch so that we can build the model, (the structure of the net is resnet35/50 locked)

* fastai can warp quickly, `flip_vert=True` often make it perform better for planet images

Smallersize the image to get faster experiment and training  

Segmentation need to label every pixel, apropriate batch size match the GPU.
* Much faster, better generalized -- **progressive resizing**, smaller size -- 64*64 not work well, but others?
* training loss should be lower than validation loss, OR IT IS UNDER FITTING. If under fitting, we can train longer, lower learing rate, or decrease regularization.
* learning rate anealing: make the learning rate go up and down. Help to explore the flap part.
NEXT PART: data augmentation, dropout, regularization, weight decay

* Avoiding running out of GPU ram, we can use the mixed recision training, `to_fp16()` in `fastai` and new CUDA driver
* Combination of matrix multiplication and unlinear activation functions, universal approximation theorem
* remember platform.ai to label the data
