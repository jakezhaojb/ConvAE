require 'cutorch'
require 'cunn'
require 'nnx'
require 'FFTconv'
require 'xlua'
require 'optim'
require 'jz'

-- This file involves all kinds of configurations!!!
--
cutorch.setDevice(2)
debugFlag = false
torch.setdefaulttensortype('torch.FloatTensor')

dataset = 'cifar'
trSize = 60000 -- for MNIST
if debugFlag then
   trSize = 1000
end

filterSize = 9
nOutplane = 128
poolSize = 8
l1weight = 1 -- To be tunned

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
   path_to_training = '/home/jz1672/Projects/what-where/TrAE/mnist/train_32x32.t7'
   path_to_testing = '/home/jz1672/Projects/what-where/TrAE/mnist/test_32x32.t7'

elseif dataset == 'cifar' then
   nInplane = 3
   path_to_training = '/home/jz1672/Data/cifar_train.t7'
   path_to_testing = '/home/jz1672/Data/cifar_test.t7'
else
   print("No dataset is found.")
   sys.exit()
end

stackFlag = true
-- TODO
