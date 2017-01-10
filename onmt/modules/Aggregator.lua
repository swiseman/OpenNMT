-- local allEncStates, allCtxs = {}, {}
-- for j = 1, batch.sourceInput:size(1) do
--     batch:setInputRow(j)
--     local encStates, context = model["encoder" .. j]:forward(batch)
--     table.insert(allEncStates, encStates)
--     table.insert(allCtxs, context)
-- end
-- batch:setInputRow(nil) -- for sanity
-- --local encStates, context = model.encoder:forward(batch)
-- local aggEncStates, catCtxs = model.aggregator:forward(allEncStates, allCtxs)
-- local decOutputs = model.decoder:forward(batch, aggEncStates, catCtxs)
-- local encGradStatesOut, gradContext, loss = model.decoder:backward(batch, decOutputs, criterion)
-- local allEncGradOuts, gradCatCtx = model.aggregator:backward(encGradStatesOut, gradContext)
-- for j = 1, batch.sourceInput:size(1) do
--     batch:setInputRow(j)
--     model["encoder" .. j]:backward(batch, allEncGradOuts[j], gradCatCtx[j])
-- end
-- batch:setInputRow(nil)

local Aggregator, parent = torch.class('onmt.Aggregator', 'nn.Container')

function Aggregator:__init(nRows)
  parent.__init(self)

  self.nRows = nRows
  -- have stuff for both cells and hiddens
  self.cellNet = self:_buildModel(nRows, encDim, decDim)
  self.hidNet = self:_buildModel(nRows, encDim, decDim)
  self:add(self.cellNet)
  self:add(self.hidNet)
  self.layerClones = {} -- use same transformation for every layer
  self.catCtx = torch.Tensor()
end

function Aggregator:_buildModel()
    return nn.Sequential()
            :add(nn.JoinTable(2))
            :add(nn.Linear(nRows*encDim, decDim))
end

-- allEncStates is an nRows-length table containing nLayers-length tables;
-- allCtxs is an nRows-length table containing batchSize x srcLen x dim tensors
function Aggregator:forward(allEncStates, allCtxs)
    -- first concatenate all the contexts
    local firstCtx = allCtxs[1]
    local rowLen = firstCtx:size(2) -- assumed constant for all rows
    self.catCtx:resize(firstCtx:size(1), self.nRows*rowLen, firstCtx:size(3))
    -- just copy
    for b = 1, self.catCtx:size(1) do -- loop over batch size
        for j = 1, self.nRows do
            self.catCtx[b]:sub((j-1)*rowLen + 1, j*rowLen):copy(allCtxs[b])
        end
    end

    -- now do aggregation
    if self.train then
        self.layInputs = {}
    end
    local aggEncStates = {}
    for i = 1, #allEncStates[1] do
        if not self.layerClones[i] then
            if i%2 == 1 then
                self.layerClones[i] = self.cellNet:clone('weight', 'gradWeight', 'bias', 'gradBias')
            else
                self.layerClones[i] = self.hidNet:clone('weight', 'gradWeight', 'bias', 'gradBias')
            end
        end
        -- get all the stuff we're concatenating
        local layInput = {}
        for j = 1, self.nRows do
            table.insert(layInput, allEncStates[j][i])
        end
        if self.train then
            table.insert(self.layInputs, layInput)
        end
        table.insert(aggEncStates, self.layerClones[i]:forward(layInput))
    end

    return aggEncStates, self.catCtx
end

-- encGradStatesOut is an nLayers-length table;
-- gradContext sho
function Aggregator:backward(encGradStatesOut, gradContext)
    local allEncGradOuts = {}
    for j = 1, self.nRows do
        allEncGradOuts[j] = {}
    end
    for i = 1, #encGradStatesOut do
        local gradIns = self.layerClones[i]:backward(self.layInputs[i], encGradStatesOut[i])
        for j = 1, self.nRows do
            table.insert(allEncGradOuts[j], gradIns[j])
        end
    end
    return allEncGradOuts
end
