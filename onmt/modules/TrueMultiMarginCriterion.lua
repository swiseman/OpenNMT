require 'nn'

local TrueMultiMarginCriterion, parent = torch.class('nn.TrueMultiMarginCriterion', 'nn.Criterion')

function TrueMultiMarginCriterion:__init(margin, ignoreIdx)
   parent.__init(self)
   self.margin = margin or 1
   self.sizeAverage = true
   self.ignoreIdx = ignoreIdx
end

function TrueMultiMarginCriterion:updateOutput(input, y)
    if not self.maxes then
        self.maxes = torch.Tensor():typeAs(input)
        self.argmaxes = torch.type(input) == 'torch.CudaTensor'
            and torch.CudaLongTensor() or torch.LongTensor()
        self.truescores = torch.Tensor():typeAs(input)
        if self.ignoreIdx then
            self.ycopy = torch.Tensor():typeAs(input)
            self.mask = torch.Tensor():typeAs(input)
        end
    end
    self.maxes:resize(input:size(1), 1)
    self.argmaxes:resize(input:size(1), 1)
    self.truescores:resize(input:size(1), 1)

    if self.ignoreIdx then
        self.ycopy:resize(y:size(1)):copy(y)
        self.mask:resizeAs(self.ycopy)
        self.mask:ne(self.ycopy, self.ignoreIdx)
    end

    local y2d = y:view(input:size(1), 1)
    self.truescores:gather(input, 2, y2d)
    -- get maxes/argmaxes of wrong things
    input:scatter(2, y2d, -math.huge)
    torch.max(self.maxes, self.argmaxes, input, 2)
    -- put original scores batck
    input:scatter(2, y2d, self.truescores)
    self.maxes:add(self.margin)
    self.maxes:add(-1, self.truescores)
    self.maxes:cmax(0)
    if self.ignoreIdx then
        self.maxes:cmul(self.mask)
    end

    self.output = self.maxes:sum()
    if self.sizeAverage then
        self.output = self.output/input:size(1)
    end
    return self.output
end

function TrueMultiMarginCriterion:updateGradInput(input, y)
    self.gradInput:resizeAs(input):zero()
    -- self.maxes should already have positive or zero values
    self.maxes:mul(1e9)
    -- if we had positive vals < 1e-9 then this is wrong, but should be fine
    self.maxes:cmin(1)

    local grads = self.maxes

    if self.sizeAverage then
        grads:div(input:size(1))
    end
    self.gradInput:scatter(2, self.argmaxes:view(input:size(1), 1), grads)
    grads:neg()
    self.gradInput:scatter(2, y:view(input:size(1), 1), grads)
    return self.gradInput
end
