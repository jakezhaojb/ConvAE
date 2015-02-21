--require 'cutorch'
--require 'cunn'
require 'myaenn'
require 'fbcunn'
require 'cudnn'
require 'nnx'
require 'xlua'
require 'optim'
--require 'jz'

-- This file involves all kinds of configurations!!!
--
cutorch.setDevice(1)
debugFlag = false
torch.setdefaulttensortype('torch.FloatTensor')

dataset = 'cifar'
trSize = 60000 -- for MNIST
if debugFlag then
   trSize = 1000
end

filterSize = 9
nOutplane = 16
poolSize = 8
l1weight = 1 -- To be tunned
init_scale_down = 1

optimState = {
   learningRate = 0.0005,
   weightDecay = 0.00001,
   momentum = 0.9,
   learningRateDecay = 5e-4
}
batchSize = 256

dofile("./Modules/init.lua")
maxPoolFlag = true
paraTied = true

if dataset == 'mnist' then
   nInplane = 1
   path_to_training = '/misc/vlgscratch3/LecunGroup/jakezhao/Data/mnist/train_32x32.t7'
   path_to_testing = '/misc/vlgscratch3/LecunGroup/jakezhao/Data/mnist/test_32x32.t7'

elseif dataset == 'cifar' then
   nInplane = 3
   path_to_training = '/misc/vlgscratch3/LecunGroup/jakezhao/Data/cifar/CIFAR_CN_train.t7'
   path_to_testing = '/misc/vlgscratch3/LecunGroup/jakezhao/Data/cifar/CIFAR_CN_test.t7'
else
   print("No dataset is found.")
   sys.exit()
end

stackFlag = true
-- TODO

poolBeta = 0.1
