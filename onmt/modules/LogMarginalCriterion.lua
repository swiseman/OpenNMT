local LogMarginalCriterion, parent = torch.class('nn.LogMarginalCriterion', 'nn.Criterion')

function LogMarginalCriterion:__init(K)
   parent.__init(self)
   self.sizeAverage = true
   self.K = K  -- number of on elements per row
end


--[[ Compute the negative of the log marginal over true targets

Parameters:

  * `input` - an NxV tensor of log-probabilities.
  * `target` - a table; target[1] is an NxV binary mask with target[1]:sum(2) == K*torch.ones(N).
                        target[2] is an NxK binary mask to allow variable numbers of true targets

--]]
function LogMarginalCriterion:updateOutput(input, target)
    if not self.buf then
        self.buf = torch.Tensor():typeAs(input)
        self.rowsums = torch.Tensor():typeAs(input)
        self.gradInput:typeAs(input)
    end
    local N, V, K = input:size(1), input:size(2), self.K
    self.buf:resize(N*K)
    self.rowsums:resize(N, 1)

    local vmask, kmask = target[1], target[2]

    -- get log probs for each row
    self.buf:maskedSelect(input, vmask)
    local logprobs = self.buf:view(N, K)

    -- sum-exp; we don't do the logsumexp trick, since we probably don't have
    -- to worry about underflow (and overflow won't happen since
    -- everything is negative)
    logprobs:exp()
    -- zero out anything that shouldn't actually be included
    logprobs:cmul(kmask)
    self.rowsums:sum(logprobs, 2)
    self.rowsums:log()
    self.output = -self.rowsums:sum() -- minimizing nll...
    if self.sizeAverage then
        self.output = self.output/N
    end

   return self.output
end


function LogMarginalCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input):zero()

    local N, V, K = input:size(1), input:size(2), self.K
    self.buf:resize(N*K)
    self.rowsums:resize(N, 1)

    local vmask, kmask = target[1], target[2]

    -- get log probs for each row
    self.buf:maskedSelect(input, vmask)
    local logprobs = self.buf:view(N, K)
    -- sum-exp
    logprobs:exp()
    logprobs:cmul(kmask)
    self.rowsums:sum(logprobs, 2)
    logprobs:cdiv(self.rowsums:expand(N, K))
    local scale = self.sizeAverage and -1/N or -1
    logprobs:mul(scale)
    self.gradInput:maskedCopy(vmask, logprobs)
    return self.gradInput
end


-- lsm = nn.LogSoftMax()
-- lmc = nn.LogMarginalCriterion(3)
-- Y = lsm:forward(torch.randn(2, 5))
-- T1 = torch.ByteTensor({{1,0,1,1,0},
--                       {0,1,1,1,0}})
-- T2 = torch.Tensor(    {{1,1,0},
--                        {1,0,0}})
-- T22 = torch.Tensor(   {{1,0,0},
--                        {1,0,0}})
-- t = torch.Tensor({1,2})
