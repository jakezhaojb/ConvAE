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

trSize = 60000 -- for MNIST
if debugFlag then
   trSize = 1000
end
filterSize = 9
nInplane = 1 -- MNIST
nOutplane = 64
poolSize = 2 
l1weight = 0.1 -- To be tunned

path_to_training = '/home/jz1672/Projects/what-where/TrAE/mnist/train_32x32.t7'
path_to_testing = '/home/jz1672/Projects/what-where/TrAE/mnist/test_32x32.t7'

optimState = {
   learningRate = 0.0005,
   weightDecay = 0.00001,
   momentum = 0.9,
   learningRateDecay = 5e-4
}
batchSize = 256

dofile("./Modules/init.lua")
maxPoolFlag = true
