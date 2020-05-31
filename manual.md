# Super Easy User Guide  
## Testing (after clonding repository):   
1. Binary classification:  
    > cd Classification
    > python test.py -pre \[model name\]
Available pre-trained models in Classification/models_6
2. Binary segmentation: run Binary_Segmentation/train_new.py  
    > cd Binary_Segmentation
    > python train_new.py -pre \[model name\] -sv \[1: saves generated test masks in ./output/, 0: doesn't save masks\]
Available pre-trained models in Classification/models_6
3. Multi-class segmentation: run Binary_Segmentation/train_new.py  
    > cd Binary_Segmentation
    > python train_new.py -pre \[model name\] -sv \[1: saves generated test masks in ./output/, 0: doesn't save masks\]
Available pre-trained models in Classification/models_6
