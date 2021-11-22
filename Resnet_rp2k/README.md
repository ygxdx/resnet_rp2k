## What we used
- rp2k dataset 
- SEResnet 50
- Image preprocessing and parallelism achieved by Mindspore package

## Files description
- [datset.py](dataset.py) <br>
    Applies image preprocessing operations from mindspore library <br>
- [resnetv2.py](resnetv2.py) <br>
    Defines implementation of Resnet architecture <br>
- [train.py](train.py) <br>
    Driver code for training the model <br>
- [eval.py](eval.py) <br>
    Evaluation script for evaluating an particular model checkpoint <br>
  
## Recreation
**Steps for training the model:**
```bash
$python train.py

usage: train.py [-h] [-bsize BATCHSIZE] [-repeatNum REPEATNUM] [-dir SAVEDIR]
               [-e EPOCH] [-opt OPTIMIZER] [-lr LEARNINGRATE] [-m MOMENTUM]
               [-wDecay WEIGHTDECAY]

optional arguments:
  -h, --help            show this help message and exit
  -bsize BATCHSIZE, --batchsize BATCHSIZE
                        batch size
  -repeatNum REPEATNUM, --repeatNum REPEATNUM
                        repeat num
  -dir SAVEDIR, --savedir SAVEDIR
                        save directory
  -e EPOCH, --epoch EPOCH
                        no of epochs
  -opt OPTIMIZER, --optimizer OPTIMIZER
                        optimizer
  -lr LEARNINGRATE, --learningRate LEARNINGRATE
                        learning rate
  -m MOMENTUM, --momentum MOMENTUM
                        momentum
  -wDecay WEIGHTDECAY, --weightDecay WEIGHTDECAY
                        optimizer

```

**Steps for evaulating the model:**
```bash
$python eval.py

usage: eval.py [-h] [-loc CHECK_POINT]

optional arguments:
  -h, --help            show this help message and exit
  -loc CHECK_POINT, --check_point CHECK_POINT
                        Model checkpoint file


```
    
## Results

| Model Architecture  | Accuracy |
| ------------- | ------------- |
| SEResnet-50  | 89.07  |


