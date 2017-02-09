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
function EnDecoder:__init(inputNetwork, rnn, generator, inputFeed)
  self.rnn = rnn
  self.inputNet = inputNetwork

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

  assert(false) -- make sure call shareScorers after cuda'ing and before flattening
  --self.goldScorer = self:_buildScorer()
  self.sampleScorer = self.goldScorer:clone('weight', 'gradWeight', 'bias', 'gradBias')

  --self:add(self.goldScorer)
  self:add(self.sampleScorer)

  self:resetPreallocation()
end


function EnDecoder:shareScorers()
    self.sampleScorer:share(self.goldScorer, 'weight', 'gradWeight', 'bias', 'gradBias')
end

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
  self.allInputs = torch.range(1, self.inputNet.vocabSize)

  self.scoreSumProto = torch.Tensor()

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end


function EnDecoder:_buildScorer()
    local mlp = nn.Sequential()
                  :add(nn.JoinTable(2))
                  :add(nn.Linear(3*self.args.rnnSize, 3/2*self.args.rnnSize))
                  :add(nn.ReLU())
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

  self.goldScorer = self:_buildScorer()
  table.insert(outputs, scorer(attnOutput))


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
function EnDecoder:forwardOne(input, prevStates, context, prevOut, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `buildNetwork('dec')`).
  onmt.utils.Table.append(inputs, prevStates)
  table.insert(inputs, input)
  table.insert(inputs, context)
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

  local outputs = self:net(t):forward(inputs)

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

function EnDecoder:forwardAndApply(batch, encoderStates, context, func)
  -- TODO: Make this a private method.

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(#encoderStates,
                                                         self.stateProto,
                                                         { batch.size, self.args.rnnSize })
  end

  local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)

  local prevOut, scores

  for t = 1, batch.targetLength do
    prevOut, states, scores = self:forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
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

  local outputs = {}
  local scoreSum = self.scoreSumProto
  scoreSum:resize(batch.size, 1):zero()

  self:forwardAndApply(batch, encoderStates, context, function (out)
    table.insert(outputs, out, scores)
    scoreSum:add(scores)
  end)

  return outputs, scoreSum
end

function EnDecoder:sampleFwd(t, stepsBack, batch, context)
    if not self.maxes then
        self.maxes = torch.CudaTensor(1)
        self.argmaxes = torch.CudaLongTensor(1)
    end

    -- get previous shit
    local outputs = self:net(t-1).output
    local numOutputs = #outputs
    local out = outputs[numOutputs-1]

    -- for saving argmaxes
    local sampleInput = self.sampleInputProto
    sampleInput:resize(stepsBack+1, batch.size)

    local prevStates = {}
    local tempNet = self:net(batch.targetLength+1)

    -- just gonna do one example at a time (probably don't wanna do it all at once for mem reasons anyway)
    for b = 1, batch.size do
        for i = 1, numOutputs-2 do
            prevStates[i] = outputs[i]:sub(b, b):expand(self.inputNet.vocabSize, outputs[i]:size(2))
        end
        local bout = out:sub(b, b):expand(self.inputNet.vocabSize, out:size(2))

        for s = 1, stepsBack+1 do
            -- find argmax shit
            local inputs = {}
            onmt.utils.Table.append(inputs, prevStates)
            table.insert(inputs, self.allInputs) -- maybe modify lstm to just take in lut embeddings
            table.insert(inputs, context)

            if self.args.inputFeed then
                table.insert(inputs, bout)
            end

            -- get max
            local currOutputs = tempNet:forward(inputs)
            torch.max(self.maxes, self.argmaxes, currOutputs:view(-1), 1)
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

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function EnDecoder:backward(batch, outputs, criterion)
    if self.gradOutputsProto == nil then
        self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers + 1,
                                                              self.gradOutputProto,
                                                              { batch.size, self.args.rnnSize })
        self.gradSampleOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers + 1,
                                                            self.gradSampleOutputProto,
                                                            { batch.size, self.args.rnnSize })
    end


    local gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                             { batch.size, self.args.rnnSize })
    local gradSampleStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradSampleOutputsProto,
                                                            { batch.size, self.args.rnnSize })
    local gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                         { batch.size, batch.sourceLength, self.args.rnnSize })

    local loss = 0

    assert(false) -- just need targetInputs, not Outputs, and they need to be 1-longer so we get energy of EOS token
    local maxBack = 2 -- sample at most 3 contiguous tokens for now
    local t = batch.targetLength
    while t >= 2 do -- first is always SOS
        local stepsBack = torch.random(0, math.min(maxBack, t-2))
        -- get states for samples
        local sampleInputs = sampleFwd(t, stepsBack)
        -- go fwd to get sampled states (and scores)


        -- assuming for now we only get loss at merge pts (and end)
        local goldScores = self.goldScorer:forward(outputs[t])
        local sampleScores = self.sampleScorer:forward(sampleStates)
        loss = loss + criterion:forward(goldScores, sampleScores)
        local goldScoreGradOut = criterion:backward(goldScores, sampleScores)
        for j = 1, #goldScoreGradOut do
            goldScoreGradOut[j]:div(batch.totalSize) -- ehhhh
        end
        local goldRNNGradOut = self.goldScorer:backward(outputs[t], goldScoreGradOut)
        goldScoreGradOut:neg()
        local sampleRNNGradOut = self.sampleScorer:backward(outputs[t], goldScoreGradOut)
        gradStatesInput[#gradStatesInput]:add(goldRNNGradOut)
        gradSampleStatesInput[#gradSampleStatesInput]:add(sampleRNNGradOut)

        for s = t, t-stepsBack, -1 do
            local gradInput = self:net(s):backward(self.inputs[s], gradStatesInput)
            local sampleIdx = maxBack-t+s+1
            local gradSampleInput = self:sampleNet(sampleIdx):backward(sampleInputs[sampleIdx], gradSampleStatesInput)
            gradContextInput:add(gradInput[self.args.inputIndex.context])
            gradContextInput:add(gradSampleInput[self.args.inputIndex.context])
            gradStatesInput[#gradStatesInput]:zero()
            gradSampleStatesInput[#gradSampleStatesInput]:zero()
            if self.args.inputFeed then -- t must be  > 1
                gradStatesInput[#gradStatesInput]:add(gradInput[self.args.inputIndex.inputFeed])
                gradSampleStatesInput[#gradSampleStatesInput]:add(gradSampleInput[self.args.inputIndex.inputFeed])
            end
            for i = 1, #self.statesProto do
                gradStatesInput[i]:copy(gradInput[i])
                gradSampleStatesInput[i]:copy(gradSampleInput[i])
            end
        end

        -- accumulate into gold state
        for i = 1, #gradStatesInput do
            gradStatesInput[i]:add(gradSampleStatesInput[i])
            gradSampleStatesInput[i]:zero()
        end

        -- move to step preceding sampled sequence
        t = t - stepsBack - 1
    end

    assert(t == 1)
    -- there is no output loss at first timestep and we must've merged, so just go back
    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)
    gradContextInput:add(gradInput[self.args.inputIndex.context])
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
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local pred = self.generator:forward(out)
    local output = batch:getTargetOutput(t)
    loss = loss + criterion:forward(pred, output)
  end)

  return loss
end


--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function EnDecoder:computeScore(batch, encoderStates, context)
  encoderStates = encoderStates
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
