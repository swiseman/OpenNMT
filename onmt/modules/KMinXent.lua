require 'nn'

local KMinXent, parent = torch.class('nn.KMinXent', 'nn.Criterion')

-- will square the distance for p=2, tho maybe we shouldn't...
function KMinXent:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.net = nn.Sequential()
                :add(nn.MM(false, true)) -- batchSize x numPreds x M
                :add(nn.Max(3))          -- batchSize x numPreds
                -- check if View(-1) is faster...
                :add(nn.Sum(2))          -- batchSize; doesn't seem like we can sum over everything at once
                :add(nn.Sum())           -- 1
                :add(nn.MulConstant(-1)) -- 1
   self.netGradOut = torch.ones(1) -- could rid of MulConstant and just make this negative
end

-- input is batchSize x numPreds x sum[outVocabSizes], where each dist is log normalized.
-- target is binary batchsize x M x sum[outVocabSizes], where target[b][m] is concatenation of 1 hot vectors.
-- loss: - sum_k max_m \sum_j ln q^(j)(m_j)  = sum_k min_m \sum_j xent(q^(j), m_j)
function KMinXent:updateOutput(input, target)
    if self.sizeAverage then
        self.net:get(5).constant_scalar = -1/input:size(1)
    else
        self.net:get(5).constant_scalar = -1
    end
    self.output = self.net:forward({input, target})
    return self.output
end


function KMinXent:updateGradInput(input, target)
    self.net:backward({input, target}, self.netGradOut)
    self.gradInput = self.net.gradInput[1]
    return self.gradInput
end


-- torch.manualSeed(2)
-- local M = 5
-- local dim = 5
-- local K = 3
--
-- crit = nn.KMinXent(2)
-- --crit = nn.KMinXent(1)
--
-- X = torch.randn(2, K*dim)
--
-- Y = torch.randn(2, M, dim)
--
--
-- crit:forward(X, Y)
-- gradIn, gradTarg = crit:backward(X, Y)
-- gradIn = gradIn:clone()
-- gradTarg = gradTarg:clone()
--
-- local eps = 1e-5
--
--
-- local function getLoss()
--     return crit:forward(X, Y)
-- end
--
-- print("X")
-- for i = 1, X:size(1) do
--     for j = 1, X:size(2) do
--         X[i][j] = X[i][j] + eps
--         local rloss = getLoss()
--         X[i][j] = X[i][j] - 2*eps
--         local lloss = getLoss()
--         local fd = (rloss - lloss)/(2*eps)
--         print(gradIn[i][j], fd)
--         X[i][j] = X[i][j] + eps
--     end
--     print("")
-- end
--
-- print("")
-- print("Y")
-- rY = Y:view(-1, dim)
-- for i = 1, rY:size(1) do
--     for j = 1, rY:size(2) do
--         rY[i][j] = rY[i][j] + eps
--         local rloss = getLoss()
--         rY[i][j] = rY[i][j] - 2*eps
--         local lloss = getLoss()
--         local fd = (rloss - lloss)/(2*eps)
--         print(gradTarg:view(-1, dim)[i][j], fd)
--         rY[i][j] = rY[i][j] + eps
--     end
--     print("")
-- end
