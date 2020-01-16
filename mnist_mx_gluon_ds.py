import numpy as np
import mxnet as mx
import random
import time

from multiprocessing import cpu_count
from mxnet import autograd as ag
from mxnet import nd
from mxnet.metric import Accuracy
from mxnet.gluon import Block, Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import Conv2D, Dense, Dropout, Flatten, MaxPool2D, HybridBlock
from mxnet.gluon.utils import split_and_load


BATCH_SIZE_PER_REPLICA = 512
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1
NUM_CLASSES = 10
EPOCHS = 10


class Model(HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = Conv2D(32, (3, 3))
            self.conv2 = Conv2D(64, (3, 3))
            self.pool = MaxPool2D(pool_size=(2, 2))
            self.dropout1 = Dropout(0.25)
            self.flatten = Flatten()
            self.dense1 = Dense(128)
            self.dropout2 = Dropout(0.5)
            self.dense2 = Dense(NUM_CLASSES)

    def hybrid_forward(self, F, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)


def data_loader(train, batch_size, num_workers):
    dataset = MNIST(train=train, transform=transform)
    return DataLoader(dataset, batch_size, shuffle=train, num_workers=num_workers)


mx.random.seed(42)
random.seed(42)

train_data = data_loader(train=True, batch_size=BATCH_SIZE, num_workers=cpu_count())
test_data = data_loader(train=False, batch_size=BATCH_SIZE, num_workers=cpu_count())

model = Model()
model.hybridize(static_alloc=True, static_shape=True)

ctx = [mx.gpu()]

# optimizer
opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
opt = mx.optimizer.create('adam', **opt_params)

# Initialize parameters randomly
model.initialize(force_reinit=True, ctx=ctx)
# fetch and broadcast parameters
params = model.collect_params()
# trainer
trainer = Trainer(params=params,
                  optimizer=opt,
                  kvstore='device')
loss_fn = SoftmaxCrossEntropyLoss()
metric = Accuracy()
start = time.perf_counter()
for epoch in range(EPOCHS):
    tick = time.time()
    for i, (data, label) in enumerate(train_data):
        if i == 0:
            tick_0 = time.time()
        data = split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = split_and_load(label, ctx_list=ctx, batch_axis=0)
        output = []
        losses = []
        with ag.record():
            for x, y in zip(data, label):
                z = model(x)
                # computes softmax cross entropy loss
                l = loss_fn(z, y)
                output.append(z)
                losses.append(l)
        # backpropagate the error for one iteration.
        for l in losses:
            l.backward()
        # Update network weights
        trainer.step(BATCH_SIZE)
        # Update metric
        metric.update(label, output)
    str1 = 'Epoch [{}], Accuracy {:.4f}'.format(epoch, metric.get()[1])
    str2 = '~Samples/Sec {:.4f}'.format(BATCH_SIZE*(i+1)/(time.time()-tick_0))
    print('%s  %s' % (str1, str2))
    metric.reset()

elapsed = time.perf_counter() - start
print('elapsed: {:0.3f}'.format(elapsed))

# use Accuracy as the evaluation metric
metric = Accuracy()
for data, label in test_data:
    data = split_and_load(data, ctx_list=ctx, batch_axis=0)
    label = split_and_load(label, ctx_list=ctx, batch_axis=0)
    outputs = []
    for x in data:
        outputs.append(model(x))
    metric.update(label, outputs)
print('validation %s=%f' % metric.get())
