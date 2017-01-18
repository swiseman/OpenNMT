local MarginalNLLCriterion, parent = torch.class('nn.MarginalNLLCriterion', 'nn.Criterion')

function MarginalNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end


--[[ This will output the negative log marginal, even though we'll ignore the log when doing gradients

Parameters:

  * `input` - an NxV tensor of probabilities.
  * `target` - a mask with 0s for probabilities to be ignored and positive numbers for probabilities to be added

--]]
function MarginalNLLCriterion:updateOutput(input, target)
    if not self.buf then
        self.buf = torch.Tensor():typeAs(input)
        self.gradInput:typeAs(input)
    end
    self.buf:resizeAs(input)
    self.buf:cmul(input, target)
    self.output = -math.log(self.buf:sum())
    if self.sizeAverage then
        self.output = self.output/input:size(1)
    end

    return self.output
end

--[[
  Gradient of negative SUM of probabilities (not negative log of sum of probabilities)
]]
function MarginalNLLCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(target)
    local scale = self.sizeAverage and -1/input:size(1) or -1
    self.gradInput:mul(scale)
    return self.gradInput
end
