-- torch test
-- cutorch test

require 'cunn'
require 'nnx'

dofile("MaxPoolUnpool.lua")

x = torch.rand(2,2,4,4):cuda()

a = nn.MaxPoolUnpool(2,2)

print(x)

y = a:forward(x)
print(y)

grad = torch.rand(y:size()):cuda()

z = a:backward(x, grad)

print(z)
print(grad)

