print '==> loading dataset'

local loaded = torch.load(path_to_training, 'ascii')
images = loaded.data:clone()
images = images:type('torch.FloatTensor')
images:div(255)
images = images[{ {1, trSize}, {}, {}, {}  }]

collectgarbage()
