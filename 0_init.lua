require 'cutorch'
require 'cunn'
require 'nnx'
require 'FFTconv'
require 'xlua'
require 'optim'
require 'jz'

-- This file involves all kinds of configurations!!!
--
cutorch.setDevice(1)
debugFlag = true

trSize = 60000 -- for MNIST
if debugFlag then
   trSize = 100
end
filterSize = 9
nInplane = 1 -- MNIST
nOutplane = 20
poolSize = 2 
l1weight = 0.01 -- To be tunned

path_to_training = '/home/jz1672/Projects/what-where/TrAE/mnist/train_32x32.t7'
path_to_testing = '/home/jz1672/Projects/what-where/TrAE/mnist/test_32x32.t7'

optimState = {
   learningRate = 0.001,
   weightDecay = 0.0001,
   momentum = 0,
   learningRateDecay = 1e-7
}
batchSize = 16
