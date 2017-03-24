local LogRankingCriterion, parent = torch.class('nn.LogRankingCriterion', 'nn.Criterion')

-- loss = -log sigmoid(score_y) - log sigmoid (-score_y')
--      = log(1 + exp(-score_y)) + log(1 + exp(score_y'))
function LogRankingCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.net = nn.Sequential()
                :add(nn.ParallelTable()
                       :add(nn.Identity())
                       :add(nn.MulConstant(-1)))
                :add(nn.JoinTable(2))
                :add(nn.LogSigmoid())
                :add(nn.Sum(2))
   self.netGradOut = torch.Tensor()
end

-- y should be 1s or 0s
function LogRankingCriterion:updateOutput(input, y)
    if input[1]:dim() ~= 2 then
        input = {input[1]:view(-1, 1), input[2]:view(-1, 1)}
    end
    local netOutput = self.net:forward(input)
    netOutput:cmul(y)
    self.output = -netOutput:sum()
    if self.sizeAverage then
        self.output = self.output/y:size(1)
    end
    return self.output
end


function LogRankingCriterion:updateGradInput(input, y)
    if input[1]:dim() ~= 2 then
        input = {input[1]:view(-1, 1), input[2]:view(-1, 1)}
    end    
    self.netGradOut:resizeAs(y):fill(-1)
    self.netGradOut:cmul(y)
    if self.sizeAverage then
        self.netGradOut:div(y:size(1))
    end
    self.gradInput = self.net:backward(input, self.netGradOut)
    return self.gradInput
end
