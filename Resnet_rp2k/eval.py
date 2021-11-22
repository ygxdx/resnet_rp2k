from resnetv2 import resnet18,se_resnet50
from efficient import efficientnet_b0
from dataset import create_dataset
from mindspore.compression.quant import QuantizationAwareTraining
from mindspore.nn.optim import Adam
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import load_checkpoint
from mindspore import Model, context
from cross_entropy import CrossEntropySmoothMixup,CrossEntropySmoothMixup2

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-loc", "--check_point", default = '/media/user_data/yg/train_seresnet50_rp2k-10_10016_acc89.ckpt',
   help="Model checkpoint file")

args = vars(ap.parse_args())

num_classes = 2388
lr = 0.001
weight_decay = 1e-4

resnet = se_resnet50(num_classes)
efficient=efficientnet_b0(num_classes)

ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
# ls = CrossEntropySmoothMixup2()
opt = Adam(resnet.trainable_params(),lr, weight_decay)

# quantizer = QuantizationAwareTraining(bn_fold=False)
# quant = quantizer.quantize(resnet)

load_checkpoint(args['check_point'], net=resnet) # loading the custom trained checkpoint

eval_data = create_dataset(dataset_path='/media/user_data/yg/all/test',do_train=False) # define the test dataset


model = Model(resnet, loss_fn=ls, optimizer=opt, metrics={'acc'})

acc = model.eval(eval_data)

print('Accuracy of model is: ', acc['acc'] * 100)
