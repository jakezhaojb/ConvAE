dofile("0_init.lua")

normalize_weight = function(nn_module, convFlag)
   if convFlag == nil then
      convFlag = true
   end
   local w = nn_module.weight:clone():type('torch.FloatTensor')
   if not convFlag then
      -- F.C
      local nImg = w:size(1)
      local nSize = w:size(2)
      local nSizeSqrt = math.floor(math.sqrt(nSize))
      w = w[{ {}, {1,nSizeSqrt^2} }]
      w = w:reshape(nImg, nSizeSqrt, nSizeSqrt)
      for i = 1, nImg do
         w[i]:div(w[i]:norm())
      end
      w_norm = w:clone()
   else
      -- Convolution
      for i = 1, w:size(1) do
         for j = 1, w:size(2) do
            w[{ i, j, {}, {} }]:div(w[{ i, j, {}, {} }]:norm())
         end
      end
      w_norm = torch.squeeze(w)
   end
   return w_norm
end


function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end


function string.begins(String, Begin)
   return Begin=='' or string.sub(String,1,string.len(Begin))==Begin
end


form_weight_table = function(dirname)
   local wTb = {}
   p = io.popen('find -L ' .. dirname .. ' -maxdepth 1 -type f')
   for fl in p:lines() do
      if string.begins(fl, './Results/model_net') then
         -- It is a model file
         ml = torch.load(fl)
         wTb[fl .. '_encoder'] = normalize_weight(ml:get(1):get(2), true)
         wTb[fl .. '_decoder'] = normalize_weight(ml:get(5):get(2), true)
      end
   end
   return wTb
end


form_model_table = function(dirname)
   local modelTb = {}
   p = io.popen('find -L ' .. dirname .. ' -maxdepth 1 -type f')
   for fl in p:lines() do
      if string.begins(fl, './Results/model_net') then
         -- It is a model file
         ml = torch.load(fl)
         ml:float()
         modelTb[fl] = ml:clone()
      end
   end
   return modelTb
end


form_output_table = function(model_table)
   local outputTb = {}
   local criterion = nn.MSECriterion()
   criterion.sizeAverage = false
   criterion:cuda()
   if dataset == 'mnist' then
      local test_loaded = torch.load(path_to_testing, 'ascii')
      images = test_loaded.data:clone()
      images = images:type('torch.FloatTensor')
      images:div(255)
   elseif dataset == 'cifar' then
      local test_loaded = torch.load(path_to_testing)
      images = test_loaded.datacn:float():reshape(test_loaded.datacn:size(1), 3, 32, 32)
   else
      print 'No'
   end
   local teSize = images:size(1)
   local shuffle = torch.randperm(teSize)[{{1,49}}]
   imagesElem = torch.Tensor():resize(49, images:size(2), images:size(3), images:size(4)):typeAs(images)
   for i = 1, 49 do
      imagesElem[i] = images[shuffle[i]]:clone()
   end
   images = imagesElem:clone()
   imagesElem = nil
   collectgarbage()
   -- produce the output
   -- testing error and Nerr
   local file_test_error = io.open('./Results/test_error.csv', 'w')
   for k, model in pairs(model_table) do
      model:cuda()
      outputTb[k] = model:forward(images:cuda()):type('torch.FloatTensor')
      local Nf = 0
      local f = 0
      for i = 1, 49 do
         local input = images[i]:float():reshape(1, images:size(2), images:size(3), images:size(4)):cuda()
         local output = model:forward(input)
         local err = criterion:forward(output, input)
         Nf = Nf + math.sqrt(err)/input:norm()
         f = f + err
      end
      -- err
      f = f / 49
      Nf = Nf / 49
      local f_str = string.format('%.2f', f)
      local Nf_str = string.format('%.2f', Nf)
      file_test_error:write(k .. ' : ' .. Nf_str .. ',' .. f_str .. '\n')
   end
   file_test_error:close()
   return outputTb
end


visualize_weight = function(weight_table)
   if gfx ~= nil then
      for k, v in pairs(weight_table) do
         gfx.image(v, {zoom=5, legend=k})
      end
   else
      torch.save('Results/filter.t7', weight_table)
   end
end


visualize_output = function(output_table)
   if gfx ~= nil then
      for k, v in pairs(output_table) do
         gfx.image(v, {legend=k})
      end
   else
      torch.save('Results/reconstruct.t7', output_table)
   end
end


visualize_weight(form_weight_table('./Results'))
visualize_output(form_output_table(form_model_table('./Results')))
