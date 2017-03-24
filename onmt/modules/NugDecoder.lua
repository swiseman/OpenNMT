local status, module = pcall(require, 'cudnn')
cudnn = status and module or nil

local SpatialConvolution = status and cudnn.SpatialConvolution or nn.SpatialConvolution

local function make_bytenet_relu_block(d, mask, kW, dil)
    -- input assumed to be batchSize x 2d x 1 x seqLen; treating as an image
    local block = nn.Sequential()
                    :add(nn.ReLU()) -- in theory, BN before
                    -- args: nInPlane, nOutPlane, kW, kH, dW, dH, padW, padH
                    :add(SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 1 x seqLen
                    :add(nn.ReLU())
    if mask then
        block:add(nn.SpatialDilatedConvolution(d, d, kW, 1, 1, 1, (kW-1)*dil, 0, dil, 1)) -- batchSize x d x 1 x seqLen+(kW-1)*dil
        block:add(nn.Narrow(4, 1, -(kW-1)*dil - 1)) -- batchSize x d x 1 x seqLen
    else
        -- args: nInPlane, nOutPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH
        block:add(nn.SpatialDilatedConvolution(d, d, kW, 1, 1, 1, (kW-1)/2+dil-1, 0, dil, 1)) -- batchSize x d x 1 x seqLen
    end
    block:add(nn.ReLU()) -- in theory, BN before
    block:add(SpatialConvolution(d, 2*d, 1, 1, 1, 1))
    return block
end

-- this is like the stuff from the Dauphin, Fan, et al. paper
local function make_gated_block(d, mask, kW, dil, use_tanh)
    -- input assumed to be batchSize x d x 1 x seqLen
    local block = nn.Sequential()
    if mask then
        block:add(nn.SpatialDilatedConvolution(d, 2*d, kW, 1, 1, 1, (kW-1)*dil, 0, dil, 1)) -- batchSize x 2d x 1 x seqLen+(kW-1)*dil
        block:add(nn.Narrow(4, 1, -(kW-1)*dil - 1)) -- batchSize x 2d x 1 x seqLen
    else
        block:add(nn.SpatialDilatedConvolution(d, 2*d, kW, 1, 1, 1, (kW-1)/2+dil-1, 0, dil, 1)) -- batchSize x 2d x 1 x seqLen
    end

    block:add(nn.ConcatTable()
                :add(use_tanh and nn.Sequential():add(nn.Narrow(2, 1, d)):add(nn.Tanh()) or nn.Narrow(2, 1, d))
                :add(nn.Sequential():add(nn.Narrow(2, d+1, d)):add(nn.Sigmoid())))
         :add(nn.CMulTable()) -- batchSize x d x 1 x seqLen
    return block
end

local function make_res_block(block)
    local res = nn.Sequential()
                    :add(nn.ConcatTable()
                             :add(nn.Identity())
                             :add(block))
                    :add(nn.CAddTable())
    return res
end

local function embs_as_img_enc(lut)
    -- maps batchSize x seqLen -> batchSize x dim x 1 x seqLen
    local dim
    if torch.type(lut) == 'nn.LookupTable' then
        dim = lut.weight:size(2) + ctxdim
    else -- probably a WordEmbedding
        dim = lut.net.weight:size(2) + ctxdim
    end
    --local dim = lut.weight:size(2)
    local enc = nn.Sequential()
                  :add(lut) -- batchSize x seqLen x dim
                  :add(nn.Transpose({2, 3})) -- batchSize x dim x seqLen
                  :add(nn.Reshape(dim, 1, -1, true)) -- batchSize x dim x 1 x seqLen
    return enc
end

local function embs_and_neg_as_img_enc(lut)
    -- maps batchSize x 2*seqLen -> batchSize x dim x 2 x seqLen.
    -- this will do the padding before true and after fake automatically
    local dim
    if torch.type(lut) == 'nn.LookupTable' then
        dim = lut.weight:size(2)
    else -- probably a WordEmbedding
        dim = lut.net.weight:size(2)
    end
    local enc = nn.Sequential()
                  :add(lut) -- batchSize x 2*seqLen x dim
                  :add(nn.SpatialZeroPadding(0, 0, 1, 1)) -- batchSize x (1 + 2*seqLen + 1) x dim
                  :add(nn.Reshape(2, -1, dim, true)) -- batchSize x 2 x seqLen x dim
                  :add(nn.Transpose({2, 4}, {3, 4})) -- batchSize x dim x 2 x seqLen
    return enc
end

local function embs_and_neg_as_condimg_enc(lut, seqLen, ctxdim)
    -- maps batchSize x 2*seqLen -> batchSize x dim x 2 x seqLen.
    -- note that seqLen is trueSeqLen+1, where true is padded before and false padded after
    local dim
    if torch.type(lut) == 'nn.LookupTable' then
        dim = lut.weight:size(2) + ctxdim
    else -- probably a WordEmbedding
        dim = lut.net.weight:size(2) + ctxdim
    end
    --local dim = lut.weight:size(2) + ctxdim
    local replicator = nn.Replicate(2*seqLen, 2, 2)
    local enc = nn.Sequential()
                  :add(nn.ParallelTable()
                       :add(lut)                         -- batchSize x 2*seqLen x dim
                       :add(nn.Sequential()
                              :add(nn.Reshape(1,-1,true)) -- bsz x 1 x ctxdim
                              :add(replicator)            -- bsz x 1 x 2*seqLen x ctxdim
                              :add(nn.Squeeze(2))))       -- bsz x 2*seqLen x ctxdim
                  :add(nn.JoinTable(3)) -- batchSize x 2*seqLen x (dim+ctxdim)
                  :add(nn.SpatialZeroPadding(0, 0, 1, 1)) -- batchSize x (1 + 2*seqLen + 1) x dim
                  :add(nn.Reshape(2, -1, dim, true)) -- batchSize x 2 x seqLen x dim
                  :add(nn.Transpose({2, 4}, {3, 4})) -- batchSize x dim x 2 x seqLen
    return enc, replicator
end

local function make_neg_gated_block(d, kW, dil, use_tanh, dout)
    -- input assumed to be batchSize x d x 2 x seqLen
    -- output should be such that output[b][d][1][t] = output[b][d][2][t-1]
    local dout = dout or d
    local block
      = nn.Sequential()
          :add(nn.ConcatTable()
                 :add(nn.Sequential()
                        :add(nn.SpatialDilatedConvolution(d, 2*dout, kW, 1, 1, 1, (kW-1)*dil, 0, dil, 1)) -- batchSize x 2d x 2 x seqLen+(kW-1)*dil
                        :add(nn.Narrow(4, 1, -(kW-1)*dil - 1)) -- batchSize x 2d x 2 x seqLen
                        :add(nn.Narrow(3, 1, 1))) -- batchSize x 2d x 1 x seqLen
                 :add(nn.Sequential()
                        :add(nn.SpatialDilatedConvolution(d, 2*dout, kW-1, 2, 1, 1, (kW-2)*dil, 0, dil, 1)) -- batchSize x 2d x 1 x seqLen+(kW-2)*dil
                        --:add(nn.Narrow(4, dil+1, -(kW-1)*dil - 1)))) -- batchSize x 2d x 1 x seqLen
                        :add(nn.Narrow(4, 1, -(kW-2)*dil - 1)))) -- batchSize x 2d x 1 x seqLen
          :add(nn.JoinTable(3)) -- batchSize x 2d x 2 x seqLen
          :add(nn.ConcatTable()
                 :add(use_tanh and nn.Sequential():add(nn.Narrow(2, 1, dout)):add(nn.Tanh()) or nn.Narrow(2, 1, dout))
                 :add(nn.Sequential():add(nn.Narrow(2, dout+1, dout)):add(nn.Sigmoid())))
         :add(nn.CMulTable()) -- batchSize x d x 2 x seqLen
    return block
end

local function make_neg_bytenet_relu_block(d, kW, dil, dout)
    -- input assumed to be batchSize x 2d x 2 x seqLen
    local dout = dout or 2*d
    local block
      = nn.Sequential()
          :add(nn.ReLU()) -- in theory, BN before
          :add(SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 2 x seqLen
          --:add(nn.SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 2 x seqLen
          :add(nn.ReLU())
          :add(nn.ConcatTable()
                 :add(nn.Sequential()
                        :add(nn.SpatialDilatedConvolution(d, d, kW, 1, 1, 1, (kW-1)*dil, 0, dil, 1)) -- batchSize x 2d x 2 x seqLen+(kW-1)*dil
                        :add(nn.Narrow(4, 1, -(kW-1)*dil - 1)) -- batchSize x d x 2 x seqLen
                        :add(nn.Narrow(3, 1, 1))) -- batchSize x d x 1 x seqLen
                 :add(nn.Sequential()
                        :add(nn.SpatialDilatedConvolution(d, d, kW-1, 2, 1, 1, (kW-2)*dil, 0, dil, 1)) -- batchSize x 2d x 2 x seqLen+(kW-1)*dil
                        :add(nn.Narrow(4, 1, -(kW-2)*dil - 1)))) -- batchSize x d x 1 x seqLen
          :add(nn.JoinTable(3)) -- batchSize x d x 2 x seqLen
          :add(nn.ReLU()) -- in theory, BN before
          :add(SpatialConvolution(d, dout, 1, 1, 1, 1)) -- batchSize x 2d x 2 x seqLen
          --:add(nn.SpatialConvolution(d, dout, 1, 1, 1, 1)) -- batchSize x 2d x 2 x seqLen
    return block
end

local function do_block_sharing(block, bytenet)
    local cat_idx = bytenet and 4 or 1
    local oned_conv = block:get(cat_idx):get(1):get(1)
    local W1, b1 = oned_conv.weight, oned_conv.bias -- 2d x d x 1 x kW, 2d
    local kW = W1:size(4)
    local twod_conv = block:get(cat_idx):get(2):get(1)
    local W2, b2 = twod_conv.weight, twod_conv.bias -- 2d x d x 2 x (kW-1), 2d

    -- assuming considering only 1-step deviations for now
    W2:narrow(3, 1, 1):copy(W1:narrow(4, 1, kW-1))
    W2:narrow(3, 2, 1):narrow(4, 1, kW-2):zero()
    W2:narrow(3, 2, 1):select(4, kW-1):copy(W1:select(4, kW))
    b2:copy(b1)
end

local function add_block_grads(block, bytenet)
    local cat_idx = bytenet and 4 or 1
    local oned_conv = block:get(cat_idx):get(1):get(1)
    local W1, b1 = oned_conv.weight, oned_conv.bias -- 2d x d x 1 x kW, 2d
    local kW = W1:size(4)
    local twod_conv = block:get(cat_idx):get(2):get(1)
    local W2, b2 = twod_conv.weight, twod_conv.bias -- 2d x d x 2 x (kW-1), 2d

    -- assuming considering only 1-step deviations for now
    W1:narrow(4, 1, kW-1):add(W2:narrow(3, 1, 1))

    --W2:narrow(3, 2, 1):narrow(4, 1, kW-2):zero()
    W1:select(4, kW):add(W2:narrow(3, 2, 1):select(4, kW-1))
    b1:add(b2)
end


local function make_final_layer()
    -- input is batchSize x d x 2 x seqLen
    local mod = nn.Sequential()
                  :add(nn.SplitTable(3))  -- 2-table w/ tensors of size batchSize x d x seqLen
                  :add(nn.ParallelTable()
                         :add(nn.Narrow(3, 2, -1))   -- batchSize x d x seqLen-1
                         :add(nn.Narrow(3, 1, -2)))  -- batchSize x d x seqLen-1
    -- output is 2-table w/ tensors of size batchSize x d x seqLen-1
    return mod
end

-- require 'nn'
--
-- lut = nn.LookupTable(7, 4)
--
-- X = torch.LongTensor({{3, 4, 5, 2, 7, 6, 2, 4, 4, 5},
--                       {2, 5, 6, 3, 7, 5, 7, 3, 2, 4}})
--
-- X2 = torch.LongTensor({{3, 4, 5, 2, 7, 3, 4, 5, 2, 7},
--                        {2, 5, 6, 3, 7, 2, 5, 6, 3, 7}})


local NugDecoder, parent = torch.class('onmt.NugDecoder', 'onmt.Network')

function NugDecoder:__init(vocabSize, nlayers, inputSize, hiddenSize, kW,
    dropout, dilate, fair, useTanh, finalRelu)
  -- assume inputSize is embedding of source and target shit
  self.nlayers = nlayers
  self.vocabSize = vocabSize
  local dropout = dropout or 0
  self.dropout = dropout
  self.lut = onmt.WordEmbedding.new(vocabSize, inputSize, pretrainedWords, false)
  local dummySeqLen = 2
  local net, replicator = embs_and_neg_as_condimg_enc(self.lut, dummySeqLen, inputSize)
  self.replicator = replicator

  local finalRelu = (finalRelu == nil) and true or finalRelu

  local dil = 0
  local inputSize = inputSize
  local convLayers = {}
  for i = 1, nlayers do
      if dilate then
          dil = math.max(2*dil, 1)
          if dil > 16 then
              dil = 1
          end
      else
          dil = 1
      end
      local layer
      if fair then
          layer = make_neg_gated_block(inputSize, kW, dil, useTanh, hiddenSize)
      else
          layer = make_neg_bytenet_relu_block(inputSize, kW, dil, hiddenSize)
      end
      table.insert(convLayers, layer)
      net:add(make_res_block(layer))
      inputSize = hiddenSize -- assuming all layers other than first are same size
  end

  if finalRelu then
      net:add(nn.ReLU())
  end

  if dropout > 0 then
      net:add(nn.Dropout(dropout))
  end

  --assert(false) -- add final layer (preceded maybe by relu or dropout or something)

  self.convLayers = convLayers
  self.fair = fair

  parent.__init(self, net)
end

function NugDecoder:updateOutput(input)
    local twiceSeqLen = input:size(2)
    self.replicator.nfeatures = twiceSeqLen
    for i = 1, #self.convLayers do
        do_block_sharing(self.convLayers[i], not self.fair)
    end
    self.output = self.net:updateOutput(input)
    return self.output
end

function NugDecoder:accGradParameters(input, gradOutput, scale)
    self.net:accGradParameters(input, gradOutput, scale)
    for i = 1, #self.convLayers do
        add_block_grads(self.convLayers[i], not self.fair)
    end
end

-- gonna do this in the stupidest way for now
function NugDecoder:_buildInfcNetwork()
    self.infcNet = self.net:clone('weight', 'gradWeight', 'bias', 'gradBias')
    self.infcReplicator = self.infcNet:get(1):get(2):get(2)
    table.remove(self.infcNet.modules, 3) -- padding
    collectgarbage()
    self.infcNet.modules[3].batchsize[2] = 1 -- nn.Reshape(1,-1,dim,true)

    -- get rid of negative shit
    for i, mod in ipairs(self.infcNet.modules) do
        if torch.type(mod) == 'nn.Sequential' then -- a res block
            local convLayer = mod:get(1):get(2) -- second thing in res block
            for j = 1, #convLayer.modules do
                if torch.type(convLayer.modules[j]) == 'nn.ConcatTable' then
                    local posPart = convLayer.modules[j]:get(1)
                    -- remove last narrow
                    posPart.modules[#posPart.modules] = nil
                    collectgarbage()
                    convLayer.modules[j] = posPart
                    -- get rid of join
                    table.remove(convLayer.modules, j+1)
                    collectgarbage()
                    break
                end
            end
        end
    end
end

-- again this is the slow way to do this
function NugDecoder:inference(samples, batch, ctx, goldPfxs)
    self.inputBuf = self.inputBuf or batch.targetInput.new()
    self.ctxBuf = self.ctxBuf or ctx.new()
    self.predBuf = self.predBuf or batch.targetInput.new()
    self.maxes = self.maxes or ctx.new()
    self.argmaxes = self.argmaxes or batch.targetInput.new()

    local inputBuf = self.inputBuf
    inputBuf:resize(samples*batch.size, batch.targetLength)
    inputBuf:select(2, 1):fill(onmt.Constants.BOS)

    local ctxBuf = self.ctxBuf
    ctxBuf:resize(sample*batch.size, ctx:size(2))
    for b = 1, batch.size do
        torch.repeatTensor(ctxBuf:sub((b-1)*samples+1, b*samples), ctx:sub(b, b), samples, 1)
    end

    local predBuf = self.predBuf
    predBuf:resizeAs(batch.targetInput:t())

    self.maxes:resize(batch.size(1), 1)
    self.argmaxes:resize(batch.size(1), 1)

    if goldPfxs then
        self.yrank = self.yrank or ctx.new()
        self.yrank:resize(batch.size, batch.targetLength):fill(1)
    end
    local agreementSteps = 0

    local randIdxs = torch.Tensor(self.vocabSize)

    for t = 2, T do
        randIdxs:randperm(self.vocabSize)
        for b = 1, batch.size do
            inputBuf:sub((b-1)*samples+1, b*samples, t, t):copy(randIdxs:sub(1, samples))
        end
        self.infcReplicator.nfeatures = t
        local scores = self.infcNet:forward({inputBuf:narrow(2, 1, t), ctxBuf})
        torch.max(self.maxes, self.argmaxes, scores:view(batch.size, samples), 2)
        for b = 1, batch.size do
            predBuf[b][t] = randIdxs[self.argmaxes[b][1]]
            if goldPfxs then
                inputBuf:sub((b-1)*samples+1, b*samples, t, t):fill(batch.targetInput[t][b])
                if predBuf[b][t] == batch.targetInput[t][b] then
                    self.yrank[b][t] = 0
                    agreementSteps = agreementSteps + 1
                end
            else
                inputBuf:sub((b-1)*samples+1, b*samples, t, t):fill(predBuf[b][t])
            end
        end
    end
    return predBuf, self.yrank, agreementSteps
end
