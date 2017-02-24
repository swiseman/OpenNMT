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

-- adapted for torch resnet code

local function make_bytenet_relu_block(d, mask, kW, dil)
    -- input assumed to be batchSize x 2d x 1 x seqLen; treating as an image
    local block = nn.Sequential()
                    :add(nn.ReLU()) -- in theory, BN before
                    -- args: nInPlane, nOutPlane, kW, kH, dW, dH, padW, padH
                    :add(cudnn.SpatialConvolution(2*d, d, 1, 1, 1, 1)) -- batchSize x d x 1 x seqLen
                    :add(nn.ReLU())
    if mask then
        assert(false)
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
        assert(false)
    else
        block:add(nn.SpatialDilatedConvolution(d, 2*d, kW, 1, 1, 1, (kW-1)/2+dil-1, 0, dil, 1)) -- batchSize x 2d x 1 x seqLen
    end

    block:add(nn.ConcatTable()
                :add(use_tanh and nn.Sequential():add(nn.Narrow(1, 1, d)):add(nn.Tanh()) or nn.Narrow(1, 1, d))
                :add(nn.Sequential():add(nn.Narrow(1, d+1, d)):add(nn.Sigmoid())))
         :add(nn.CMulTable()) -- batchSize x 2d x 1 x seqLen
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


return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
