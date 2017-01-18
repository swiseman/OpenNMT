-- require 'nn'
-- onmt = {}

local MarginalNLLCriterion, parent = torch.class('onmt.MarginalNLLCriterion', 'nn.Criterion')

function MarginalNLLCriterion:__init(ignoreIdx)
   parent.__init(self)
   self.sizeAverage = true
   self.ignoreIdx = ignoreIdx
end


--[[ This will output the negative log marginal, even though we'll ignore the log when doing gradients

Parameters:

  * `input` - an NxV tensor of probabilities.
  * `target` - a mask with 0s for probabilities to be ignored and positive numbers for probabilities to be added

--]]
function MarginalNLLCriterion:updateOutput(input, target)
    if not self.buf then
        self.buf = torch.Tensor():typeAs(input)
        self.rowSums = torch.Tensor():typeAs(input)
        self.gradInput:typeAs(input)
    end
    self.buf:resizeAs(input)
    self.buf:cmul(input, target)
    self.rowSums:resize(input:size(1), 1)
    if self.ignoreIdx then
        self.buf:select(2, self.ignoreIdx):zero()
    end
    self.rowSums:sum(self.buf, 2) -- will store for backward
    -- use buf
    local logRowSums = self.buf:narrow(2, 1, 1)
    logRowSums:log(self.rowSums)
    self.output = -logRowSums:sum()
    if self.sizeAverage then
        self.output = self.output/input:size(1)
    end

    return self.output
end

--[[
  Respecting the log probably doesn't actually matter that much....
]]
function MarginalNLLCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(target)
    if self.ignoreIdx then
        self.gradInput:select(2, self.ignoreIdx):zero()
    end
    self.gradInput:cdiv(self.rowSums:expand(input:size(1), input:size(2)))
    local scale = self.sizeAverage and -1/input:size(1) or -1
    self.gradInput:mul(scale)
    return self.gradInput
end


-- torch.manualSeed(2)
-- local mlp = nn.Sequential()
--          :add(nn.Linear(4,5))
--          :add(nn.SoftMax())
--
-- local crit = onmt.MarginalNLLCriterion()
-- crit.sizeAverage = false
--
-- local X = torch.randn(2, 4)
-- local T = torch.zeros(2, 5)
-- T[1][1] = 1
-- T[1][3] = 1
-- T[2][2] = 1
--
-- mlp:zeroGradParameters()
-- mlp:forward(X)
-- crit:forward(mlp.output, T)
-- local gradOut = crit:backward(mlp.output, T)
-- mlp:backward(X, gradOut)
--
-- local eps = 1e-5
--
-- local function getLoss()
--     mlp:forward(X)
--     return crit:forward(mlp.output, T)
-- end
--
-- local W = mlp:get(1).weight
-- for i = 1, W:size(1) do
--     for j = 1, W:size(2) do
--         W[i][j] = W[i][j] + eps
--         local rloss = getLoss()
--         W[i][j] = W[i][j] - 2*eps
--         local lloss = getLoss()
--         local fd = (rloss - lloss)/(2*eps)
--         print(mlp:get(1).gradWeight[i][j], fd)
--         W[i][j] = W[i][j] + eps
--     end
-- end
