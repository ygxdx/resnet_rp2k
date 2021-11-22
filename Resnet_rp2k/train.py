#!/usr/bin/env python
# coding: utf-8


# importing necessary modules and functions
import os
import random
import argparse
# from preprocess import create_dataset
from dataset import create_dataset
from mindspore import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits,FocalLoss
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim import Adam,RMSProp
from mindspore.nn.optim import SGD
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from resnetv2 import resnet18,se_resnet50,resnet101
from efficient import efficientnet_b0
import numpy as np
from mindspore.compression.quant import QuantizationAwareTraining
import argparse
from mindspore import nn
from cross_entropy import CrossEntropySmoothMixup


# for receiving arguments in CLI
ap = argparse.ArgumentParser()


ap.add_argument("-bsize", "--batchsize", default =256,
   help="batch size")
ap.add_argument("-repeatNum", "--repeatNum", default = 1,
   help="repeat num")
ap.add_argument("-dir", "--savedir", default = 'new_check/',
   help="save directory")
ap.add_argument("-e", "--epoch", default =20 ,
   help="no of epochs")
ap.add_argument("-opt", "--optimizer", default = 'Adam',
   help="optimizer")
ap.add_argument("-lr", "--learningRate", default = 0.001,
   help="learning rate")
ap.add_argument("-m", "--momentum", default = 0.9,
   help="momentum")
ap.add_argument("-wDecay", "--weightDecay", default = 1e-4,
   help="optimizer")

args = vars(ap.parse_args())



mode = 1
if mode == 1:
## PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
if mode == 2:
## Graph MODE ## Cant do since we are working with only one GPU
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", enable_graph_kernel=True)



batch_size = int(args['batchsize'])
repeat_num = int(args['repeatNum'])
num_classes = 2388



from mindspore.train.callback import Callback

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, step_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.step_per_eval = step_per_eval

    def step_end(self, run_context):
        cb_param = run_context.original_args()
        cur_step = cb_param.cur_step_num
        # if cur_epoch % self.eval_per_epoch == 0:
        #     acc = self.model.eval(self.eval_dataset)
        #     self.epoch_per_eval["step"].append(cur_epoch)
        #     self.epoch_per_eval["acc"].append(acc["acc"])
        #     print('\n',acc, '\n')
        if cb_param.cur_step_num % 1000 == 0:
            acc = self.model.eval(self.eval_dataset)
            self.step_per_eval["step"].append(cur_step)
            self.step_per_eval["acc"].append(acc["acc"])
            print('\n', acc, '\n')
            
            
class LossMonitor(Callback):
    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            if cb_params.cur_step_num % 100 == 0:
                print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss))



ckpt_save_dir = args['savedir']
eval_per_epoch = 1
epoch_size = args['epoch']
epoch_per_eval = {"step": [], "acc": []}



# net = resnet18(num_classes)
net = se_resnet50(num_classes)
# net = efficientnet_b0(num_classes)
# ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
ls = CrossEntropySmoothMixup()
lr = args['learningRate']

weight_decay = args['weightDecay']
momentum = args['momentum']
optimizer = args['optimizer']

if optimizer == 'Momentum':
    opt = Momentum(net.trainable_params(),lr, momentum)
if optimizer == 'Adam':
    opt = Adam(net.trainable_params(),lr, weight_decay)
if optimizer == 'SGD':
    opt = SGD(net.trainable_params(),lr, weight_decay)
if optimizer == 'RMS':
    opt = RMSProp(net.trainable_params(),lr, momentum,weight_decay)
    
# quantizer = QuantizationAwareTraining(bn_fold=False)
# quant = quantizer.quantize(net)

model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})

train = create_dataset(dataset_path='/media/user_data/yg/all/train',do_train=True)
eval_data = create_dataset(dataset_path='/media/user_data/yg/all/test',do_train=False)

loss_cb = LossMonitor()
time = TimeMonitor()

eval_cb = EvalCallBack(model, eval_data, eval_per_epoch, epoch_per_eval)
config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1000)
ckpoint_cb = ModelCheckpoint(prefix="train_seresnet50_rp2k", directory=ckpt_save_dir, config=config_ck)


model.train(epoch_size, train, callbacks=[ckpoint_cb, loss_cb,time])


