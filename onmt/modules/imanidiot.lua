local LogMarginalCriterion, parent = torch.class('nn.LogMarginalCriterion', 'nn.Criterion')

function LogMarginalCriterion:__init(K)
   parent.__init(self)
   self.sizeAverage = true
   self.K = K  -- number of on elements per row
end

-- input is expected to be an NxV tensor of log probabilities;
-- target is expected to be an NxV mask containing 1s for correct probs
function LogMarginalCriterion:updateOutput(input, target)
    if not self.buf then
        self.buf = torch.Tensor():typeAs(input)
        self.maxes = torch.Tensor():typeAs(input)
        self.argmaxes = torch.type(input) == 'torch.CudaTensor' and torch.CudaLongTensor() or torch.LongTensor()
        self.rowsums = torch.Tensor():typeAs(input)
    end
    local N, V, K = input:size(1), input:size(2), self.K
    self.buf:resize(N*K)
    self.maxes:resize(N, 1)
    self.argmaxes:resize(N, 1)
    self.rowsums:resize(N, 1)

    -- get log probs for each row
    self.buf:maskedSelect(input, target)
    local logprobs = self.buf:view(N, K)
    -- get row-wise maxes
    torch.max(self.maxes, self.argmaxes, logprobs, 2)
    -- subtract max
    logprobs:add(-1, self.maxes:expand(N, K))
    -- sum-exp
    logprobs:exp()
    self.rowsums:sum(logprobs, 2)
    self.rowsums:log()
    self.rowsums:add(self.maxes)
    self.output = -self.rowsums:sum() -- minimizing nll...
    if self.sizeAverage then
        self.output = self.output/K
    end

   return self.output
end


function LogMarginalCriterion:updateGradInput(input, target)
    
end
