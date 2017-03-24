
local function make_bytenet_relu_block(d, mask, kW, dil)
    -- input assumed to be batchSize x 2d x 1 x seqLen; treating as an image
    local block = nn.Sequential()
                    :add(nn.ReLU()) -- in theory, BN before
                    -- args: nInPlane, nOutPlane, kW, kH, dW, dH, padW, padH
                    :add(cudnn.SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 1 x seqLen
                    :add(nn.ReLU())
    if mask then
        block:add(nn.SpatialDilatedConvolution(d, d, kW, 1, 1, 1, (kW-1)*dil, 0, dil, 1)) -- batchSize x d x 1 x seqLen+(kW-1)*dil
        block:add(nn.Narrow(4, 1, -(kW-1)*dil - 1)) -- batchSize x d x 1 x seqLen
    else
        -- args: nInPlane, nOutPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH
        block:add(nn.SpatialDilatedConvolution(d, d, kW, 1, 1, 1, (kW-1)/2+dil-1, 0, dil, 1)) -- batchSize x d x 1 x seqLen
    end
    block:add(nn.ReLU()) -- in theory, BN before
    block:add(cudnn.SpatialConvolution(d, 2*d, 1, 1, 1, 1))
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
    local dim = lut.weight:size(2)
    local enc = nn.Sequential()
                  :add(lut) -- batchSize x seqLen x dim
                  :add(nn.Transpose({2, 3})) -- batchSize x dim x seqLen
                  :add(nn.Reshape(dim, 1, -1, true)) -- batchSize x dim x 1 x seqLen
    return enc
end

--------------------------------------------------------------------------------

local function embs_and_neg_as_img_enc(lut)
    -- maps batchSize x 2*seqLen -> batchSize x dim x 2 x seqLen.
    -- this will do the padding before true and after fake automatically
    local dim = lut.weight:size(2)
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
    local dim = lut.weight:size(2) + ctxdim
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
          :add(cudnn.SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 2 x seqLen
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
          :add(cudnn.SpatialConvolution(d, dout, 1, 1, 1, 1)) -- batchSize x 2d x 2 x seqLen
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

function NugDecoder:__init(dict, layers, inputSize, hiddenSize, kW,
    dropout, dilate, fair, useTanh)
  -- assume inputSize is embedding of source and target shit
  local dropout = dropout or 0
  self.dropout = dropout
  local lut = onmt.WordEmbedding.new(dict:size(), inputSize, pretrainedWords, false)
  local dummySeqLen = 2
  local net, replicator = embs_and_neg_as_condimg_enc(lut, dummySeqLen, inputSize)
  self.replicator = replicator

  local dil = 0
  local inputSize = inputSize
  local convLayers = {}
  for i = 1, layers do
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
      table.append(convLayers, layer)
      net:add(make_res_block(layer))
      inputSize = hiddenSize -- assuming all layers other than first are same size
  end

  assert(false) -- add final layer (preceded maybe by relu or dropout or something)

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
