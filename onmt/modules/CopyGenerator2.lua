--[[Simple CopyGenerator. Given RNN state and (unnormalized) attn scores produce categorical distribution.

--]]
local CopyGenerator2, parent = torch.class('onmt.CopyGenerator2', 'nn.Container')


function CopyGenerator2:__init(rnnSize, outputSize, tanhQuery)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize, tanhQuery)
  self:add(self.net)
end

-- N.B. this uses attnLayer, but should maybe use last real layer (in which case we need 3 inputs)
function CopyGenerator2:_buildGenerator(rnnSize, outputSize, tanhQuery)
    local tstate = nn.Identity()() -- attnlayer (numEffectiveLayers+1)
    local context = nn.Identity()()

    -- get unnormalized attn scores
    local targetT = nn.Linear(rnnSize, rnnSize)(tstate)
    if tanhQuery then
        targetT = nn.Tanh()(targetT)
    end
    local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
    attn = nn.Sum(3)(attn) -- batchL x sourceL

    -- concatenate with regular output shit
    local regularOutput = nn.Linear(rnn_size, outputSize)(tstate)
    local catDist = nn.SoftMax()(nn.JoinTable(2)({regularOutput, attn}))
    return nn.gModule({tstate, context}, {catDist})

end

function CopyGenerator2:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function CopyGenerator2:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  return self.gradInput
end

function CopyGenerator2:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
