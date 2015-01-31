dofile("0_init.lua")
dofile("1_data.lua")
dofile("2_convae.lua")
dofile("3_train.lua")

for i = 1, 50 do
   if maxPoolFlag then
      -- tune the learningRate
      if i == 1 then
         optimState.learningRate = optimState.learningRate / 10
         print("MaxPooling warning, LearningRate is REDUCED to: " .. optimState.learningRate)
      elseif i == 2 then
         optimState.learningRate = optimState.learningRate * 10
         print("MaxPooling warning, LearningRate is RECOVERD to: " .. optimState.learningRate)
      else
         -- do noting
      end
   end
   train()
end
