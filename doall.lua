dofile("0_init.lua")
dofile("1_data.lua")
dofile("2_convae.lua")
dofile("3_train.lua")

for i = 1, 30 do
   train()
end
