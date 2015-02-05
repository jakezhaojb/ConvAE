print '==> loading dataset'

if dataset == 'mnist' then
   print "==> MNIST loading."
   local loaded = torch.load(path_to_training, 'ascii')
   images = loaded.data:clone()
   images = images:type('torch.FloatTensor')
   images:div(255)
   images = images[{ {1, trSize}, {}, {}, {}  }]

elseif dataset == 'cifar' then
   print "CIFAR loading."
   local loaded = torch.load(path_to_training)
   images = loaded.datacn:reshape(50000,3,32,32)
   images = images:type('torch.FloatTensor')
   if trSize > images:size(1) then
      trSize = images:size(1)
   else
      images = images[{ {1, trSize}, {}, {}, {}  }]
   end
else
   print('wrong path to data.')
   sys.exit()
end

collectgarbage()
