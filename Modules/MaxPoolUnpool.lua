local MaxPoolUnpool, parent = torch.class('nn.MaxPoolUnpool', 'nn.Module')

function MaxPoolUnpool:__init(kW, kH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
end

function MaxPoolUnpool:updateOutput(input)
   self.poolModule = nn.SpatialMaxPooling(self.kW, self.kH)
   if input:size():size() == 4 then
      self.UpsampledModule = nn.SpatialUpSampling(self.kW, self.kH, 3, 4)
   else
      self.UpsampledModule = nn.SpatialUpSampling(self.kW, self.kH)
   end
   if input:type() == 'torch.CudaTensor' then
      self.poolModule:cuda()
   end
   pooled = self.poolModule:forward(input)
   self.output = self.UpsampledModule:forward(pooled:float()) -- TODO shit... Ugly!
   -- TODO cuda map??
   self.output:map(input:float(), function(xx, yy) if xx==yy then return xx else return 0 end end)
   self.output = self.output:cuda()
   return self.output
end

function MaxPoolUnpool:updateGradInput(input, gradOutput)
   self.output = self.output:float()
   gradOutput_ = gradOutput:clone():float()
   gradOutput_:map(self.output, function(xx, yy) if yy~=0 then return xx else return 0 end end)
   local pooledGrad = self.UpsampledModule:backward(self.poolModule.output:float(), gradOutput_):cuda()
   self.gradInput = self.poolModule:backward(input, pooledGrad)
   return self.gradInput
end
