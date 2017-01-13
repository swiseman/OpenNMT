local BoxTableEncoder, parent = torch.class('onmt.BoxTableEncoder', 'nn.Container')

function BoxTableEncoder:__init(args)
    parent.__init(self)
    self.args = args
    self.network = self:_buildModel()
    self:add(self.network)
end

-- --[[ Return a new Encoder using the serialized data `pretrained`. ]]
-- function BoxTableEncoder.load(pretrained)
--     assert(false)
--   local self = torch.factory('onmt.Encoder')()
--
--   self.args = pretrained.args
--   parent.__init(self, pretrained.modules[1])
--
--   self:resetPreallocation()
--
--   return self
-- end

--[[ Return data to serialize. ]]
function BoxTableEncoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

-- function Encoder:resetPreallocation()
--   -- Prototype for preallocated hidden and cell states.
--   self.stateProto = torch.Tensor()
--
--   -- Prototype for preallocated output gradients.
--   self.gradOutputProto = torch.Tensor()
--
--   -- Prototype for preallocated context vector.
--   self.contextProto = torch.Tensor()
-- end

function BoxTableEncoder:_buildModel()
    local args = self.args
    local x = nn.Identity()() -- batcSize*nRows*srcLen x nFeatures
    local lut = nn.LookupTable(args.vocabSize, args.encDim)
    self.lut = lut
    local featEmbs
    if args.feat_merge == "concat" then
        -- concatenates embeddings of all features and applies MLP
        featEmbs = nn.Tanh()(
          nn.Linear(args.nFeatures*args.encDim, args.encDim)(
            nn.View(-1, args.nFeatures*args.encDim)(
             lut(x))))
    else
        -- adds embeddings of all features and applies bias and nonlinearity
        -- (i.e., embeds sparse features)
        featEmbs = nn.Tanh()(
          nn.Add(args.encDim)(
            nn.Sum(2)(
              lut(x))))
    end
    -- featEmbs are batchSize*nRows*nCols x encDim

    for i = 2, args.nLayers do
        featEmbs = nn.Tanh()(nn.Linear(encDim, encDim)(featEmbs))
    end

    if args.dropout then
        featEmbs = nn.Dropout(args.dropout)(featEmbs) -- maybe don't want?
    end

    -- attn ctx should be batchSize x nRows*nCols x dim
    local ctx = nn.View(-1, args.nRows*args.nCols, args.encDim)(featEmbs)

    -- for now let's assume we also want row-wise summaries
    local byRows = nn.View(-1, args.nRows, args.nCols, args.encDim)(featEmbs)
    if args.pool == "mean" then
        byRows = nn.Mean(3)(byRows)
    else
        byRows = nn.Max(3)(byRows)
    end
    -- byRows is now batchSize x nRows x encDim
    local flattenedByRows = nn.View(-1, args.nRows*args.encDim)(byRows)

    -- finally need to make something that can be copied into an lstm
    self.transforms = {}
    local outputs = {}
    for i = 1, args.effectiveDecLayers do
        local lin = nn.Linear(args.nRows*args.encDim, args.decDim)
        table.insert(self.transforms, lin)
        table.insert(outputs, lin(flattenedByRows))
    end

    table.insert(outputs, ctx)
    local mod = nn.gModule({x}, outputs)
    -- output is a table with an encoding for each layer of the dec, followed by the ctx
    return mod
end

function BoxTableEncoder:shareTranforms()
    for i = 3, #self.transforms do
        if i % 2 == 1 then
            self.transforms[i]:share(self.transforms[1], 'weight', 'gradWeight', 'bias', 'gradBias')
        else
            self.transforms[i]:share(self.transforms[2], 'weight', 'gradWeight', 'bias', 'gradBias')
        end
    end
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states: layer-length table with batchSize x decDim tensors
  2. - context matrix H: batchSize x nRows*nCols x encDim
--]]
function BoxTableEncoder:forward(batch)
  local finalStates = self.network:forward(batch:getSource())
  local context = table.remove(finalStates) -- pops, i think
  return finalStates, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function BoxTableEncoder:backward(batch, gradStatesOutput, gradContextOutput)
    if self.args.input_feed then
        table.remove(gradStatesOutput) -- last thing is empty input feed thing
    end
    table.insert(gradStatesOutput, gradContextOutput)
    local gradInputs = self.network:backward(batch:getSource(), gradStatesOutput)
    return gradInputs
end
