require 'image'

dofile("vis.lua")

epoch_10_flag = true

os.execute('mkdir -p ./Results/images')
os.execute('mkdir -p ./Results/images/filter')
os.execute('mkdir -p ./Results/images/recons')

local save_path_filter = './Results/images/filter/'
local save_path_recons = './Results/images/recons/'

filters = torch.load('./Results/filter.t7')
recons = torch.load('./Results/reconstruct.t7')

-- for filters
local n = 1
local file_filters = io.open(save_path_filter .. 'filters_config.csv', 'w')
for k, v in pairs(filters) do
   if epoch_10_flag and k:match('epoch_10') then
      if v:size(2) ~= 1 and v:size(2) ~= 3 then
         v = v:transpose(1,2)
      end
      local Irec = image.toDisplayTensor({input=v, padding=1, nrow=7})
      n_str = string.format('%03i', n)
      image.save(save_path_filter .. n_str .. '.png', Irec)
      file_filters:write(n_str .. ' : ' .. k .. '\n')
      n = n + 1;
   end
end
file_filters:close()


-- for reconstructions
n = 1
local file_recons = io.open(save_path_recons .. 'recons_config.csv', 'w')
for k, v in pairs(recons) do
   if epoch_10_flag and k:match('epoch_10') then
      local Irec = image.toDisplayTensor({input=v, padding=1, nrow=7})
      n_str = string.format('%03i', n)
      image.save(save_path_recons .. n_str .. '.png', Irec)
      file_recons:write(n_str .. ' : ' .. k .. '\n')
      n = n + 1;
   end
end
file_recons:close()



