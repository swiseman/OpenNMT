--[[ Unit to decode a sequence of output tokens.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local EnDecoder, parent = torch.class('onmt.EnDecoder', 'onmt.Sequencer')


--[[ Construct a decoder layer.

Parameters:

  * `inputNetwork` - input nn module.
  * `rnn` - recurrent module, such as [onmt.LSTM](onmt+modules+LSTM).
  * `generator` - optional, an output [onmt.Generator](onmt+modules+Generator).
  * `inputFeed` - bool, enable input feeding.
--]]
function EnDecoder:__init(inputNetwork, rnn, inputFeed, nSampleInputs, maxBack)
  self.rnn = rnn
  self.inputNet = inputNetwork
  self.maxBack = maxBack or 2
  self.nSampleInputs = nSampleInputs

  self.args = {}
  self.args.rnnSize = self.rnn.outputSize
  self.args.numEffectiveLayers = self.rnn.numEffectiveLayers

  self.args.inputIndex = {}
  self.args.outputIndex = {}

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.args.inputFeed = inputFeed

  parent.__init(self, self:_buildModel())

  --assert(false) -- make sure call shareScorers after cuda'ing and before flattening
  --self.goldScorer = self:_buildScorer()
  --self.sampleScorer = self.goldScorer:clone('weight', 'gradWeight', 'bias', 'gradBias')

  --self:add(self.goldScorer)
  --self:add(self.sampleScorer)

  -- will use these on ctx
  self.meanPool = nn.Mean(2)
  self.maxPool = nn.Max(2)
  self:add(self.meanPool)
  self:add(self.maxPool)

  self:resetPreallocation()
end


-- function EnDecoder:shareScorers()
--     self.sampleScorer:share(self.goldScorer, 'weight', 'gradWeight', 'bias', 'gradBias')
-- end

--[[ Return a new EnDecoder using the serialized data `pretrained`. ]]
function EnDecoder.load(pretrained)
  local self = torch.factory('onmt.EnDecoder')()

  self.args = pretrained.args

  parent.__init(self, pretrained.modules[1])
  self.generator = pretrained.modules[2]
  self:add(self.generator)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function EnDecoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function EnDecoder:resetPreallocation()
  if self.args.inputFeed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()
  self.gradSampleOutputProto = torch.Tensor()

  self.sampleInputProto = torch.LongTensor()

  self.scoreSumProto = torch.Tensor()

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end


function EnDecoder:_buildScorer()
    local mlp = nn.Sequential()
                  :add(nn.JoinTable(2))
                  :add(nn.Linear(3*self.args.rnnSize, 3/2*self.args.rnnSize))
                  :add(nn.ReLU())
                  --:add(nn.Tanh())
                  :add(nn.Linear(3/2*self.args.rnnSize, 1))
    return mlp
end

--[[ Build a default one time-step of the decoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function EnDecoder:_buildModel()
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self.args.numEffectiveLayers do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)
  self.args.inputIndex.x = #inputs

  local context = nn.Identity()() -- batchSize x sourceLength x rnnSize
  table.insert(inputs, context)
  self.args.inputIndex.context = #inputs

  local meanCtx = nn.Identity()()
  table.insert(inputs, meanCtx)
  self.args.inputIndex.meanCtx = #inputs

  local maxCtx = nn.Identity()()
  table.insert(inputs, maxCtx)
  self.args.inputIndex.maxCtx = #inputs

  local inputFeed
  if self.args.inputFeed then
    inputFeed = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, inputFeed)
    self.args.inputIndex.inputFeed = #inputs
  end

  -- Compute the input network.
  local input = self.inputNet(x)

  -- If set, concatenate previous decoder output.
  if self.args.inputFeed then
    input = nn.JoinTable(2)({input, inputFeed})
  end
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)

  -- The output of a subgraph is a node: split it to access the last RNN output.
  outputs = { outputs:split(self.args.numEffectiveLayers) }

  -- Compute the attention here using h^L as query.
  local attnLayer = onmt.GlobalAttention(self.args.rnnSize) --, true)
  attnLayer.name = 'decoderAttn'
  local attnOutput = attnLayer({outputs[#outputs], context})
  if self.rnn.dropout > 0 then
    attnOutput = nn.Dropout(self.rnn.dropout)(attnOutput)
  end
  table.insert(outputs, attnOutput)

  local scorer = self:_buildScorer()
  local scores = scorer({attnOutput, meanCtx, maxCtx})

  -- local rsize = self.args.rnnSize
  -- local scores = nn.Linear(3/2*rsize, 1)(
  --                  nn.Tanh()(
  --                    nn.Linear(3*rsize, 3/2*rsize)(
  --                      nn.JoinTable(2)({attnOutput, meanCtx, maxCtx})
  --                   )))
  table.insert(outputs, scores)

  return nn.gModule(inputs, outputs)
end

--[[ Mask padding means that the attention-layer is constrained to
  give zero-weight to padding. This is done by storing a reference
  to the softmax attention-layer.

  Parameters:

  * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
--]]
function EnDecoder:maskPadding(sourceSizes, sourceLength)
  if not self.decoderAttn then
    self.network:apply(function (layer)
      if layer.name == 'decoderAttn' then
        self.decoderAttn = layer
      end
    end)
  end

  self.decoderAttn:replace(function(module)
    if module.name == 'softmaxAttn' then
      local mod
      if sourceSizes ~= nil then
        mod = onmt.MaskedSoftmax(sourceSizes, sourceLength)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmaxAttn'
      mod:type(module._type)
      self.softmaxAttn = mod
      return mod
    else
      return module
    end
  end)
end

--[[ Run one step of the decoder.

Parameters:

  * `input` - input to be passed to inputNetwork.
  * `prevStates` - stack of hidden states (batch x layers*model x rnnSize)
  * `context` - encoder output (batch x n x rnnSize)
  * `prevOut` - previous distribution (batch x #words)
  * `t` - current timestep

Returns:

 1. `out` - Top-layer hidden state.
 2. `states` - All states.
--]]
function EnDecoder:forwardOne(input, prevStates, context, meanOverStates, maxOverStates, prevOut, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `buildNetwork('dec')`).
  onmt.utils.Table.append(inputs, prevStates)
  table.insert(inputs, input)
  table.insert(inputs, context)
  table.insert(inputs, meanOverStates)
  table.insert(inputs, maxOverStates)
  local inputSize
  if torch.type(input) == 'table' then
    inputSize = input[1]:size(1)
  else
    inputSize = input:size(1)
  end

  if self.args.inputFeed then
    if prevOut == nil then
      table.insert(inputs, onmt.utils.Tensor.reuseTensor(self.inputFeedProto,
                                                         { inputSize, self.args.rnnSize }))
    else
      table.insert(inputs, prevOut)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end

  local outputs = self:net(t, true):forward(inputs)

  -- Make sure decoder always returns table.
  if type(outputs) ~= "table" then outputs = { outputs } end

  local numOutputs = #outputs
  local scores = outputs[numOutputs]
  local out = outputs[numOutputs-1]
  local states = {}
  for i = 1, numOutputs - 2 do
    table.insert(states, outputs[i])
  end

  return out, states, scores
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function EnDecoder:forwardAndApply(batch, encoderStates, context, meanOverStates, maxOverStates, func)
  -- TODO: Make this a private method.

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(#encoderStates,
                                                         self.stateProto,
                                                         { batch.size, self.args.rnnSize })
  end

  local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)

  local prevOut, scores

  for t = 1, batch.targetLength do
    prevOut, states, scores = self:forwardOne(batch:getTargetInput(t), states,
        context, meanOverStates, maxOverStates, prevOut, t)
    func(prevOut, t, scores)
  end
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - a `Batch` object.
  * `encoderStates` - a batch of initial decoder states (optional) [0]
  * `context` - the context to apply attention to.

  Returns: Table of top hidden state for each timestep.
--]]
function EnDecoder:forward(batch, encoderStates, context)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })
  if self.train then
    self.inputs = {}
  end

  --local outputs = {}
  local scoreCumSum = self.scoreSumProto
  scoreCumSum:resize(batch.targetLength, batch.size)

  -- might think about just taking beginning and end
  local meanOverStates = self.meanPool:forward(context)
  local maxOverStates = self.maxPool:forward(context)

  self:forwardAndApply(batch, encoderStates, context, meanOverStates, maxOverStates,
    function (out, t, scores)
        --table.insert(outputs, out)
        if t > 1 then
            scoreCumSum[t]:add(scoreCumSum[t-1], scores:view(-1))
        else
            scoreCumSum[t]:copy(scores:view(-1))
        end
    end)

  return scoreCumSum
end

-- this takes the max at each timestep
function EnDecoder:getNegativeSamplesRul(t, stepsBack, batch, context)
    if not self.maxes then
        self.maxes = torch.Tensor():typeAs(context):resize(1)
        self.argmaxes = torch.type(context) == 'torch.CudaTensor' and torch.CudaLongTensor(1) or torch.LongTensor(1)
        self.allInputs = torch.type(context) == 'torch.CudaTensor' and torch.range(1, self.inputNet.vocabSize):cudaLong() or torch.range(1, self.inputNet.vocabSize):long()
    end

    -- get previous shit
    local outputs = self:net(t-stepsBack-1, true).output
    local numOutputs = #outputs
    local out = outputs[numOutputs-1]

    -- for saving argmaxes
    local sampleInput = self.sampleInputProto
    sampleInput:resize(stepsBack+1, batch.size)

    local prevStates = {}
    local tempNet = self:net(batch.targetLength+1, true)

    local meanOverStates = self.meanPool.output
    local maxOverStates = self.maxPool.output

    -- just gonna do one example at a time (probably don't wanna do it all at once for mem reasons anyway)
    for b = 1, batch.size do
        for i = 1, numOutputs-2 do
            prevStates[i] = outputs[i]:sub(b, b):expand(self.inputNet.vocabSize, outputs[i]:size(2))
        end
        local bout = out:sub(b, b):expand(self.inputNet.vocabSize, out:size(2))
        local bctx = context:sub(b, b):expand(self.inputNet.vocabSize, context:size(2), context:size(3))
        local bmean = meanOverStates:sub(b, b):expand(self.inputNet.vocabSize, meanOverStates:size(2))
        local bmax = maxOverStates:sub(b, b):expand(self.inputNet.vocabSize, maxOverStates:size(2))

        for s = 1, stepsBack+1 do
            -- find argmax shit
            local inputs = {}
            onmt.utils.Table.append(inputs, prevStates)
            table.insert(inputs, self.allInputs) -- maybe modify lstm to just take in lut embeddings
            table.insert(inputs, bctx)
            table.insert(inputs, bmean)
            table.insert(inputs, bmax)

            if self.args.inputFeed then
                table.insert(inputs, bout)
            end

            -- get max
            local currOutputs = tempNet:forward(inputs)
            torch.max(self.maxes, self.argmaxes, currOutputs[numOutputs]:view(-1), 1)
            local argmax = self.argmaxes[1]
            sampleInput[s][b] = argmax

            if s < stepsBack + 1 then -- prepare for next timestep
                for i = 1, numOutputs-2 do
                    prevStates[i] = currOutputs[i]:sub(argmax, argmax)
                       :expand(self.inputNet.vocabSize, currOutputs[i]:size(2))
                end
                bout = currOutputs[numOutputs-1]:sub(argmax, argmax)
                    :expand(self.inputNet.vocabSize, currOutputs[numOutputs-1]:size(2))
            end
        end -- end for s
    end -- end for b
    return sampleInput
end

-- this samples vocabulary words and then takes the max
function EnDecoder:getNegativeSamples(t, stepsBack, batch, context)
    if not self.maxes then
        self.maxes = torch.Tensor():typeAs(context):resize(1)
        self.argmaxes = torch.type(context) == 'torch.CudaTensor' and torch.CudaLongTensor(1) or torch.LongTensor(1)
        self.randPerm = torch.Tensor(self.inputNet.vocabSize)
        self.randInputs = torch.type(context) == 'torch.CudaTensor' and torch.CudaLongTensor(self.nSampleInputs) or torch.LongTensor(self.nSampleInputs)
    end

    -- get previous shit
    local outputs = self:net(t-stepsBack-1, true).output
    local numOutputs = #outputs
    local out = outputs[numOutputs-1]

    -- for saving argmaxes
    local sampleInput = self.sampleInputProto
    sampleInput:resize(stepsBack+1, batch.size)

    local prevStates = {}
    local tempNet = self:net(batch.targetLength+1, true)

    local meanOverStates = self.meanPool.output
    local maxOverStates = self.maxPool.output

    -- just gonna do one example at a time for now
    for b = 1, batch.size do

        for i = 1, numOutputs-2 do
            prevStates[i] = outputs[i]:sub(b, b):expand(self.nSampleInputs, outputs[i]:size(2))
        end
        local bout = out:sub(b, b):expand(self.nSampleInputs, out:size(2))
        local bctx = context:sub(b, b):expand(self.nSampleInputs, context:size(2), context:size(3))
        local bmean = meanOverStates:sub(b, b):expand(self.nSampleInputs, meanOverStates:size(2))
        local bmax = maxOverStates:sub(b, b):expand(self.nSampleInputs, maxOverStates:size(2))

        for s = 1, stepsBack+1 do
            -- sample things to max over (for now once per batch)
            self.randPerm:randperm(self.inputNet.vocabSize)
            self.randInputs:copy(self.randPerm:sub(1, self.nSampleInputs))

            -- find argmax shit
            local inputs = {}
            onmt.utils.Table.append(inputs, prevStates)
            table.insert(inputs, self.randInputs)
            table.insert(inputs, bctx)
            table.insert(inputs, bmean)
            table.insert(inputs, bmax)

            if self.args.inputFeed then
                table.insert(inputs, bout)
            end

            -- get max
            local currOutputs = tempNet:forward(inputs)
            torch.max(self.maxes, self.argmaxes, currOutputs[numOutputs]:view(-1), 1)
            local argmax = self.argmaxes[1]
            sampleInput[s][b] = self.randInputs[argmax]

            if s < stepsBack + 1 then -- prepare for next timestep
                for i = 1, numOutputs-2 do
                    prevStates[i] = currOutputs[i]:sub(argmax, argmax)
                       :expand(self.inputNet.vocabSize, currOutputs[i]:size(2))
                end
                bout = currOutputs[numOutputs-1]:sub(argmax, argmax)
                    :expand(self.inputNet.vocabSize, currOutputs[numOutputs-1]:size(2))
            end
        end -- end for s
    end -- end for b
    return sampleInput
end


function EnDecoder:fwdOnSamples(t, stepsBack, batch, context, sampleInputs, pfxScore)
    if not self.sampleScores then
        self.sampleScores = torch.Tensor():typeAs(context)
    end
    self.sampleScores:resize(batch.size):copy(pfxScore)

    -- get previous shit
    local outputs = self:net(t-stepsBack-1, true).output
    local numOutputs = #outputs
    local prevOut = outputs[numOutputs-1]

    local prevStates = {}
    for i = 1, numOutputs-2 do
        table.insert(prevStates, outputs[i])
    end

    local meanOverStates = self.meanPool.output
    local maxOverStates = self.maxPool.output

    local scores

    for s = 1, stepsBack+1 do
        -- should automatically use net:(batch.targetLength+s) and save inputs in self.inputs[batch.targetLength+s]
        prevOut, prevStates, scores = self:forwardOne(sampleInputs[s], prevStates, context, meanOverStates,
            maxOverStates, prevOut, batch.targetLength+s)
        self.sampleScores:add(scores)
    end

    return self.sampleScores
end

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function EnDecoder:backward(batch, cumSums, criterion)
    local nOutLayers = self.args.numEffectiveLayers + 2
    local laySizes = {}
    for i = 1, nOutLayers-1 do
        table.insert(laySizes, {batch.size, self.args.rnnSize})
    end
    -- for scores
    table.insert(laySizes, {batch.size, 1})

    if self.gradOutputsProto == nil then
        self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(nOutLayers,
                                                              self.gradOutputProto,
                                                              laySizes)
                                                              --{ batch.size, self.args.rnnSize })

        self.gradSampleOutputsProto = onmt.utils.Tensor.initTensorTable(nOutLayers,
                                                            self.gradSampleOutputProto,
                                                            laySizes)

        self.meanPoolOutputsProto = torch.Tensor():typeAs(self.gradOutputsProto[1])
        self.maxPoolOutputsProto = torch.Tensor():typeAs(self.gradOutputsProto[1])
        self.y = torch.Tensor():typeAs(self.gradOutputsProto[1])
    end


    local gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                             laySizes)
    local gradSampleStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradSampleOutputsProto,
                                                            laySizes)
    local gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                         { batch.size, batch.sourceLength, self.args.rnnSize })

    local meanPoolGradOut = self.meanPoolOutputsProto:resize(batch.size, self.args.rnnSize):zero()
    local maxPoolGradOut = self.maxPoolOutputsProto:resize(batch.size, self.args.rnnSize):zero()
    local y = self.y:resize(batch.size):fill(1)

    local loss = 0
    local context = self.inputs[1][self.args.inputIndex.context]

    local maxBack = self.maxBack -- sample at most 3 contiguous tokens for now
    local t = batch.targetLength
    while t >= 2 do -- first is always SOS
        --torch.manualSeed(t)
        local stepsBack = torch.random(0, math.min(maxBack, t-2))
        --local stepsBack = 0
        -- get indices of negative samples
        local sampleInputs = self:getNegativeSamples(t, stepsBack, batch, context)
        -- go fwd to get sampled states (and scores); pass in score of pfx before sampling
        local sampleScores = self:fwdOnSamples(t, stepsBack, batch, context, sampleInputs, cumSums[t-stepsBack-1])
        loss = loss + criterion:forward({cumSums[t], sampleScores}, y)
        local scoreGradOuts = criterion:backward({cumSums[t], sampleScores}, y)
        for j = 1, #scoreGradOuts do
            scoreGradOuts[j]:div(batch.totalSize) -- ehhhh
        end

        -- N.B. these score gradOuts don't change, since we add scores of each timestep
        gradStatesInput[nOutLayers]:copy(scoreGradOuts[1])
        gradSampleStatesInput[nOutLayers]:copy(scoreGradOuts[2])
        --gradSampleStatesInput[nOutLayers]:neg()

        for s = t, t-stepsBack, -1 do
            --print(self.inputs[s])
            --print(self:net(s).modules[14]:get(4).gradWeight)
            local gradInput = self:net(s):backward(self.inputs[s], gradStatesInput)
            --print(self:net(s).modules[14]:get(4).gradWeight)
            local sampleIdx = batch.targetLength + stepsBack-t+s+1
            --print(self.inputs[sampleIdx])
            --print(self:net(sampleIdx).modules[14]:get(4).gradWeight)
            local gradSampleInput = self:net(sampleIdx):backward(self.inputs[sampleIdx], gradSampleStatesInput)
            --print(self:net(t).modules[14]:get(4).gradWeight)
            gradContextInput:add(gradInput[self.args.inputIndex.context])
            gradContextInput:add(gradSampleInput[self.args.inputIndex.context])
            meanPoolGradOut:add(gradInput[self.args.inputIndex.meanCtx])
            meanPoolGradOut:add(gradSampleInput[self.args.inputIndex.meanCtx])
            maxPoolGradOut:add(gradInput[self.args.inputIndex.maxCtx])
            maxPoolGradOut:add(gradSampleInput[self.args.inputIndex.maxCtx])
            gradStatesInput[nOutLayers-1]:zero()
            gradSampleStatesInput[nOutLayers-1]:zero()
            if self.args.inputFeed then -- t must be  > 1
                gradStatesInput[nOutLayers-1]:add(gradInput[self.args.inputIndex.inputFeed])
                gradSampleStatesInput[nOutLayers-1]:add(gradSampleInput[self.args.inputIndex.inputFeed])
            end
            for i = 1, #self.statesProto do
                gradStatesInput[i]:copy(gradInput[i])
                gradSampleStatesInput[i]:copy(gradSampleInput[i])
            end
        end

        -- accumulate into gold state
        for i = 1, nOutLayers-1 do
            gradStatesInput[i]:add(gradSampleStatesInput[i])
            gradSampleStatesInput[i]:zero()
        end

        -- move to step preceding sampled sequence
        t = t - stepsBack - 1
        --break
    end

    --assert(t == 1)
    -- we've just merged all gradOuts
    gradStatesInput[nOutLayers]:zero() -- no output loss at first timestep
    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)
    gradContextInput:add(gradInput[self.args.inputIndex.context])
    meanPoolGradOut:add(gradInput[self.args.inputIndex.meanCtx])
    maxPoolGradOut:add(gradInput[self.args.inputIndex.maxCtx])

    -- actually go backward on pooling
    gradContextInput:add(self.meanPool:backward(context, meanPoolGradOut))
    gradContextInput:add(self.maxPool:backward(context, maxPoolGradOut))

    for i = 1, #self.statesProto do
        gradStatesInput[i]:copy(gradInput[i])
    end

    return gradStatesInput, gradContextInput, loss
end

--[[ Compute the loss on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function EnDecoder:computeLoss(batch, encoderStates, context, criterion)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local scoreCumSum = self.scoreSumProto
  scoreCumSum:resize(batch.targetLength, batch.size)

  local meanOverStates = self.meanPool:forward(context)
  local maxOverStates = self.maxPool:forward(context)

  local loss = 0
  self:forwardAndApply(batch, encoderStates, context, meanOverStates, maxOverStates,
  function (out, t, scores)
      --table.insert(outputs, out)
      if t > 1 then
          scoreCumSum[t]:add(scoreCumSum[t-1], scores:view(-1))
      else
          scoreCumSum[t]:copy(scores:view(-1))
      end
  end)

  local y = self.y:resize(batch.size):fill(1)
  local maxBack = self.maxBack
  local t = batch.targetLength

  while t >= 2 do -- first is always SOS
      --torch.manualSeed(t)
      local stepsBack = torch.random(0, math.min(maxBack, t-2))
      --local stepsBack = 0
      -- get indices of negative samples
      local sampleInputs = self:getNegativeSamples(t, stepsBack, batch, context)
      -- go fwd to get sampled states (and scores)
      local sampleScores = self:fwdOnSamples(t, stepsBack, batch, context, sampleInputs, scoreCumSum[t-stepsBack-1])
      loss = loss + criterion:forward({scoreCumSum[t], sampleScores}, y)
      -- move to step preceding sampled sequence
      t = t - stepsBack - 1
      --break
  end

  return loss
end


--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function EnDecoder:computeScore(batch, encoderStates, context)
    assert(false)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local score = {}

  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local pred = self.generator:forward(out)
    for b = 1, batch.size do
      if t <= batch.targetSize[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
      end
    end
  end)

  return score
end
