-- Return effective embeddings size based on user options.
local function resolveEmbSizes(opt, dicts, wordSizes)
  local wordEmbSize
  local featEmbSizes = {}

  wordSizes = onmt.utils.String.split(wordSizes, ',')

  if opt.word_vec_size > 0 then
    wordEmbSize = opt.word_vec_size
  else
    wordEmbSize = tonumber(wordSizes[1])
  end

  for i = 1, #dicts.features do
    local size

    if i + 1 <= #wordSizes then
      size = tonumber(wordSizes[i + 1])
    elseif opt.feat_merge == 'sum' then
      size = opt.feat_vec_size
    else
      size = math.floor(dicts.features[i]:size() ^ opt.feat_vec_exponent)
    end

    table.insert(featEmbSizes, size)
  end

  return wordEmbSize, featEmbSizes
end

local function buildInputNetwork(opt, dicts, wordSizes, pretrainedWords, fixWords)
  local wordEmbSize, featEmbSizes = resolveEmbSizes(opt, dicts, wordSizes)

  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               wordEmbSize,
                                               pretrainedWords,
                                               fixWords)

  local inputs
  local inputSize = wordEmbSize

  local multiInputs = #dicts.features > 0

  if multiInputs then
    inputs = nn.ParallelTable()
      :add(wordEmbedding)
  else
    inputs = wordEmbedding
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    local vocabSizes = {}
    for i = 1, #dicts.features do
      table.insert(vocabSizes, dicts.features[i]:size())
    end

    local featEmbedding = onmt.FeaturesEmbedding.new(vocabSizes, featEmbSizes, opt.feat_merge)
    inputs:add(featEmbedding)
    inputSize = inputSize + featEmbedding.outputSize
  end

  local inputNetwork

  if multiInputs then
    inputNetwork = nn.Sequential()
      :add(inputs)
      :add(nn.JoinTable(2))
  else
    inputNetwork = inputs
  end

  return inputNetwork, inputSize
end

local function buildEncoder(opt, dicts)
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.src_word_vec_size,
                                                    opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  if opt.brnn then
    -- Compute rnn hidden size depending on hidden states merge action.
    local rnnSize = opt.rnn_size
    if opt.brnn_merge == 'concat' then
      if opt.rnn_size % 2 ~= 0 then
        error('in concat mode, rnn_size must be divisible by 2')
      end
      rnnSize = rnnSize / 2
    elseif opt.brnn_merge == 'sum' then
      rnnSize = rnnSize
    else
      error('invalid merge action ' .. opt.brnn_merge)
    end

    local rnn = RNN.new(opt.layers, inputSize, rnnSize, opt.dropout, opt.residual)

    return onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
  else
    local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

    return onmt.Encoder.new(inputNetwork, rnn)
  end
end

local function buildDecoder(opt, dicts, verbose)
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.tgt_word_vec_size,
                                                    opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local RNN = onmt.LSTM
  if opt.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  local generator

  if #dicts.features > 0 then
    generator = onmt.FeaturesGenerator.new(opt.rnn_size, dicts.words:size(), dicts.features)
  else
    generator = onmt.Generator.new(opt.rnn_size, dicts.words:size(), opt.margin > 0)
  end

  if opt.input_feed == 1 then
    if verbose then
      _G.logger:info(" * using input feeding")
    end
    inputSize = inputSize + opt.rnn_size
  end

  local rnn = RNN.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)
  if opt.en then
      return onmt.EnDecoder.new(inputNetwork, rnn, opt.input_feed == 1, opt.num_sample_inputs,
         opt.max_back, opt.sample_scheme)
  else
      return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
  end
end

local function loadEncoder(pretrained, clone)
  local brnn = #pretrained.modules == 2

  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

local function loadDecoder(pretrained, clone)
  if clone then
    pretrained = onmt.utils.Tensor.deepClone(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end


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

function embs_and_neg_as_img_enc(lut)
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

function embs_and_neg_as_condimg_enc(lut, seqLen, ctxdim)
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

function make_neg_gated_block(d, kW, dil, use_tanh)
    -- input assumed to be batchSize x d x 2 x seqLen
    -- output should be such that output[b][d][1][t] = output[b][d][2][t-1]
    local block
      = nn.Sequential()
          :add(nn.ConcatTable()
                 :add(nn.Sequential()
                        :add(nn.SpatialDilatedConvolution(d, 2*d, kW, 1, 1, 1, (kW-1)*dil, 0, dil, 1)) -- batchSize x 2d x 2 x seqLen+(kW-1)*dil
                        :add(nn.Narrow(4, 1, -(kW-1)*dil - 1)) -- batchSize x 2d x 2 x seqLen
                        :add(nn.Narrow(3, 1, 1))) -- batchSize x 2d x 1 x seqLen
                 :add(nn.Sequential()
                        :add(nn.SpatialDilatedConvolution(d, 2*d, kW-1, 2, 1, 1, (kW-2)*dil, 0, dil, 1)) -- batchSize x 2d x 1 x seqLen+(kW-2)*dil
                        --:add(nn.Narrow(4, dil+1, -(kW-1)*dil - 1)))) -- batchSize x 2d x 1 x seqLen
                        :add(nn.Narrow(4, 1, -(kW-2)*dil - 1)))) -- batchSize x 2d x 1 x seqLen
          :add(nn.JoinTable(3)) -- batchSize x 2d x 2 x seqLen
          :add(nn.ConcatTable()
                 :add(use_tanh and nn.Sequential():add(nn.Narrow(2, 1, d)):add(nn.Tanh()) or nn.Narrow(2, 1, d))
                 :add(nn.Sequential():add(nn.Narrow(2, d+1, d)):add(nn.Sigmoid())))
         :add(nn.CMulTable()) -- batchSize x d x 2 x seqLen
    return block
end

function make_neg_bytenet_relu_block(d, kW, dil)
    -- input assumed to be batchSize x 2d x 2 x seqLen
    local block
      = nn.Sequential()
          :add(nn.ReLU()) -- in theory, BN before
          --:add(cudnn.SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 2 x seqLen
          :add(nn.SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 2 x seqLen
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
          --:add(cudnn.SpatialConvolution(d, 2*d, 1, 1, 1, 1)) -- batchSize x 2d x 2 x seqLen
          :add(nn.SpatialConvolution(d, 2*d, 1, 1, 1, 1)) -- batchSize x 2d x 2 x seqLen
    return block
end

function do_block_sharing(block, bytenet)
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

function make_final_layer()
    -- input is batchSize x d x 2 x seqLen
    local mod = nn.Sequential()
                  :add(nn.SplitTable(3))  -- 2-table w/ tensors of size batchSize x d x seqLen
                  :add(nn.ParallelTable()
                         :add(nn.Narrow(3, 2, -1))   -- batchSize x d x seqLen-1
                         :add(nn.Narrow(3, 1, -2)))  -- batchSize x d x seqLen-1
    -- output is 2-table w/ tensors of size batchSize x d x seqLen-1
    return mod
end

require 'nn'

lut = nn.LookupTable(7, 4)
-- let pad=1
--lut.weight[1]:zero()

-- X = torch.LongTensor({{1, 3, 4, 5, 2, 7, 6, 2, 4, 4, 5, 1},
--                       {1, 2, 5, 6, 3, 7, 5, 7, 3, 2, 4, 1}})
--
-- X2 = torch.LongTensor({{1, 3, 4, 5, 2, 7, 3, 4, 5, 2, 7, 1},
--                        {1, 2, 5, 6, 3, 7, 2, 5, 6, 3, 7, 1}})

X = torch.LongTensor({{3, 4, 5, 2, 7, 6, 2, 4, 4, 5},
                      {2, 5, 6, 3, 7, 5, 7, 3, 2, 4}})

X2 = torch.LongTensor({{3, 4, 5, 2, 7, 3, 4, 5, 2, 7},
                       {2, 5, 6, 3, 7, 2, 5, 6, 3, 7}})

return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
