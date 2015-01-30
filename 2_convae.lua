-- Convolutional Autoencoder
-- Framework1: valid + pad1 + valid or valid + pad2 + valid -> both are only capable to reconstruct the central part of input
-- Framework2: same + same -> capable to reconstruct the whole image; same = pad1 + valid


conv_autoencoder = function()

   local pad2 = filterSize - 1

   local encoder = nn.Sequential()
   encoder:add(nn.SpatialPadding(pad2, pad2, pad2, pad2, 3, 4))
   encoder:add(nn.SpatialConvolutionFFT(nInplane, nOutplane, filterSize, filterSize))
   encoder:add(nn.ReLU())

   local decoder = nn.Sequential()
   decoder:add(nn.SpatialPadding(pad2, pad2, pad2, pad2, 3, 4))
   decoder:add(nn.SpatialConvolutionFFT(nOutplane, nInplane, filterSize, filterSize))
   decoder:add(nn.ReLU())

   -- Tie the weights
   conv_filter = torch.rand(encoder:get(2).weight:size()):typeAs(encoder:get(2).weight)
   encoder:get(2).weight = conv_filter
   decoder:get(2).weight = conv_filter:transpose(1,2) -- TODO tied!!!

   -- Remark: no need to flip the weights

   local conv_ae = nn.Sequential()
   conv_ae:add(encoder)
   conv_ae:add(jz.SpatialMaxPoolingPos(poolSize, poolSize))
   conv_ae:add(jz.SpatialMaxUnpoolingPos(poolSize, poolSize))
   conv_ae:add(nn.L1Penalty(l1weight))
   conv_ae:add(decoder)

   local criterion = nn.MSECriterion()
   criterion.sizeAverage = false

   conv_ae:cuda()
   criterion:cuda()

   return conv_ae, criterion

end

model, criterion = conv_autoencoder()

