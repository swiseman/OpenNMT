--[[ Default decoder generator. Given RNN state, produce categorical distribution.

Simply implements $$softmax(W h + b)$$.
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')


function Generator:__init(rnnSize, outputSize, margin)
  parent.__init(self, self:_buildGenerator(rnnSize, outputSize, margin))
end

function Generator:_buildGenerator(rnnSize, outputSize, margin)
  local mod = nn.Sequential()
    :add(nn.Linear(rnnSize, outputSize))
  if not margin then
    mod:add(nn:LogSoftMax())
  end
  return mod
end

function Generator:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
