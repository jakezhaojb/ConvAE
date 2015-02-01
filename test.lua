require 'cunn' 
require 'jz' 
require 'image' 

path_to_training = '/home/jz1672/Projects/what-where/TrAE/mnist/train_32x32.t7'

trSize = 100

local loaded = torch.load(path_to_training, 'ascii')
images = loaded.data:clone()
images = images:type('torch.FloatTensor')
images:div(255)
images = images[{ {1, trSize}, {}, {}, {}  }]

poolSize = 2

model = nn.Sequential()
model:add(jz.SpatialMaxPoolingPos(poolSize, poolSize))
model:add(jz.SpatialMaxUnpoolingPos(poolSize, poolSize))

y = model:forward(images):type('torch.DoubleTensor')

torch.save('test_input.t7', images)
torch.save('test_output.t7', y)
