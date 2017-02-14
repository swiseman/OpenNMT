require('nngraph')

--[[ Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

--]]
local CPGlobalAttention, parent = torch.class('onmt.CPGlobalAttention', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function CPGlobalAttention:__init(dim, justConcat)
  parent.__init(self, self:_buildModel(dim, justConcat))
  self.dim = dim
end

function CPGlobalAttention:setRepeats(batchSize, nInputs)
    self.targetViewer:resetSize(batchSize, -1, self.dim)
    self.attnScoreViewer:resetSize(batchSize*nInputs, -1)
    self.attnDistViewer:resetSize(batchSize, nInputs, -1)
end

function CPGlobalAttention:shareParams(net)
    local myfound = 0
    local mylin1, mylin2
    -- first find our own linears
    self.net:apply(function(mod)
                        if mod.name then
                            if mod.name == "targetTlin" then
                                mylin1 = mod
                                myfound = myfound + 1
                            elseif mod.name == "ccLin" then
                                mylin2 = mod
                                myfound = myfound + 1
                            end
                        end
                    end)
    assert(myfound == 2)

    local otherfound = 0
    local otherlin1, otherlin2
    net:apply(function(mod)
                        if mod.name then
                            if mod.name == "targetTlin" then
                                otherlin1 = mod
                                otherfound = otherfound + 1
                            elseif mod.name == "ccLin" then
                                otherlin2 = mod
                                otherfound = otherfound + 1
                            end
                        end
                    end)
    assert(otherfound == 2)
    mylin1:share(otherlin1, 'weight', 'bias')
    mylin2:share(otherlin2, 'weight', 'bias')
end

function CPGlobalAttention:_buildModel(dim, justConcat)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local targetTlin = nn.Linear(dim, dim, false)
  targetTlin.name = "targetTlin"
  local targetT = targetTlin(inputs[1]) -- batchL x dim
  local context = inputs[2] -- batchL x sourceTimesteps x dim; 2nd and 3rd dim will get transposed

  -- Get attention.
  local defaultBatchSize, defaultNInpts = 32, 10
  self.targetViewer = nn.View(defaultBatchSize, -1, dim)
  local targetsByBatch = self.targetViewer(targetT) -- batchSize x nInputs x dim
  local attn = nn.MM(false, true)({targetsByBatch, context}) -- batchL x nInputs x sourceTimesteps
  --local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  --attn = nn.Sum(3)(attn)
  self.attnScoreViewer = nn.View(defaultBatchSize*defaultNInpts, -1) -- need 2d thing for softmax to work
  attn = self.attnScoreViewer(attn) -- batchL*nInputs x sourceTimesteps

  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)

  self.attnDistViewer = nn.View(defaultBatchSize, defaultNInpts, -1) -- back to batchL x nInputs x sourceTimesteps
  --attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL
  attn = self.attnDistViewer(attn)

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x nInputs x dim --batchL x 1 x dim
  --contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.View(-1, dim)(contextCombined) -- batchL*nInputs x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2

  local ccLin = nn.Linear(dim*2, dim, false)
  ccLin.name = "ccLin"
  local contextOutput = nn.Tanh()(ccLin(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
