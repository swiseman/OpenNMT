--require 'nn'

local KMinDist, parent = torch.class('nn.KMinDist', 'nn.Criterion')

-- will square the distance for p=2, tho maybe we shouldn't...
function KMinDist:__init(p, maxBatchSize, maxK)
   parent.__init(self)
   self.sizeAverage = true
   self.p = p or 2
   assert(self.p == 1 or self.p == 2)
   local maxBatchSize = maxBatchSize or 1024
   local maxK = maxK or 3
   self.range = torch.range(0, maxBatchSize*maxK-1)
end

-- input is batchsize x K*dim; target is batchsize x M x dim
-- \sum_k min_m dist(input_k, target_m)
function KMinDist:updateOutput(input, target)
    local bsz, dim, M, K = input:size(1), target:size(3), target:size(2), input:size(2)/target:size(3)
    self.diff = self.diff or input.new()
    self.sums = self.sums or input.new()
    self.mins = self.mins or input.new()
    if not self.argmins then
        self.argmins = torch.type(self.mins) == "torch.CudaTensor"
            and torch.CudaLongTensor() or torch.LongTensor()
    end
    self.diff:resize(bsz, K, M, dim)
    self.sums:resize(bsz, K, M, 1)
    self.mins:resize(bsz, K, 1)
    self.argmins:resize(bsz, K, 1)

    local diff, sums = self.diff, self.sums
    diff:add(input:view(bsz, K, 1, dim):expand(bsz, K, M, dim),
          -1, target:view(bsz, 1, M, dim):expand(bsz, K, M, dim))
    if self.p == 1 then
        diff:abs()
    else -- p == 2
        diff:pow(2)
    end
    sums:sum(diff, 4)
    -- if self.p == 2 then
    --     sums:sqrt()
    -- end
    torch.min(self.mins, self.argmins, sums:squeeze(4), 3)
    self.output = self.mins:sum()

    if self.p == 2 then
        self.output = self.output/2
    end

    if self.sizeAverage then
        self.output = self.output/bsz
    end

    return self.output
end

function KMinDist:updateGradInput(input, target)
    local bsz, dim, M, K = input:size(1), target:size(3), target:size(2), input:size(2)/target:size(3)
    self.gradInput:resizeAs(input)
    self.diff:resize(bsz, K, M, dim)
    local diff = self.diff
    -- could really save this from fwd pass if we double the memory
    diff:add(input:view(bsz, K, 1, dim):expand(bsz, K, M, dim),
          -1, target:view(bsz, 1, M, dim):expand(bsz, K, M, dim))

    -- recalculate argmins so we can index into a 2d tensor
    self.newIdxs = self.newIdxs or self.argmins.new()
    local newIdxs = self.newIdxs
    newIdxs:resize(bsz*K):copy(self.range:sub(1, bsz*K)):mul(M)
    newIdxs:add(self.argmins:view(-1))
    self.gradInput:view(-1, dim):index(diff:view(-1, dim), 1, newIdxs) -- holds (input_k - target_m)

    if self.p == 1 then
        self.gradInput:sign()
    end

    if self.sizeAverage then
        self.gradInput:div(bsz)
    end

    return self.gradInput

end


-- torch.manualSeed(2)
-- local M = 5
-- local dim = 5
-- local K = 3
-- local mlp = nn.Sequential()
--          :add(nn.Linear(4, K*dim))
--
-- --crit = nn.KMinDist(2)
-- crit = nn.KMinDist(1)
--
-- X = torch.randn(2, 4)
--
-- Y = torch.randn(2, M, dim)
--
--
-- mlp:zeroGradParameters()
-- mlp:forward(X)
-- print("loss", crit:forward(mlp.output, Y))
-- local gradOut = crit:backward(mlp.output, Y)
-- print("gradOut", gradOut)
-- mlp:backward(X, gradOut)
--
-- local eps = 1e-5
--
-- local function getLoss()
--     mlp:forward(X)
--     return crit:forward(mlp.output, Y)
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
--     print("")
-- end
