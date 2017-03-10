local function buildInputNetwork(opt, dicts, pretrainedWords, fixWords)
  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               opt.word_vec_size,
                                               pretrainedWords,
                                               fixWords)

  local inputs
  local inputSize = opt.word_vec_size

  local multiInputs = #dicts.features > 0

  if multiInputs then
    inputs = nn.ParallelTable()
      :add(wordEmbedding)
  else
    inputs = wordEmbedding
  end

  -- Sequence with features.
  if #dicts.features > 0 then
    local featEmbedding = onmt.FeaturesEmbedding.new(dicts.features,
                                                     opt.feat_vec_exponent,
                                                     opt.feat_vec_size,
                                                     opt.feat_merge)
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

  -- this just assumes we have posn feats
local function buildInputNetworkWithPosns(opt, dicts, pretrainedWords, fixWords,
     nRows, nCols)
  local wordEmbedding = onmt.WordEmbedding.new(dicts.words:size(), -- vocab size
                                               opt.word_vec_size,
                                               pretrainedWords,
                                               fixWords)

  if opt.feat_merge == "sum" then
      opt.feat_vec_size = opt.word_vec_size
  end
  local rowLut = nn.LookupTable(nRows, opt.feat_vec_size)
  local colLut = nn.LookupTable(nCols, opt.feat_vec_size)

  local inputs = nn.Sequential()
     :add(nn.ParallelTable()
       :add(wordEmbedding)
       :add(rowLut)
       :add(colLut))

  local inputSize
  if opt.feat_merge == "sum" then
      inputs:add(nn.CAddTable())
      inputSize = opt.feat_vec_size
  else
      inputs:add(nn.JoinTable(2))
      inputSize = opt.word_vec_size + 2*opt.feat_vec_size
  end

  return inputs, inputSize
end

local function buildEncoder(opt, dicts, nRows, nCols)
  local inputNetwork, inputSize
  if nRows then
      inputNetwork, inputSize = buildInputNetworkWithPosns(opt, dicts,
          opt.pre_word_vecs_enc, opt.fix_word_vecs_enc, nRows, nCols)
  else
      inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)
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

    local rnn = onmt.LSTM.new(opt.layers, inputSize, rnnSize, opt.dropout, opt.residual)

    return onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
  else
    local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

    return onmt.Encoder.new(inputNetwork, rnn)
  end
end


local function buildRec(opt, tripV)

    local nDiscFeatures = 3 -- triples
    assert(false) -- need to hand this to decoder
    local recViewer = nn.View(1, -1, opt.rnn_size)
    local mod = nn.Sequential()
                 :add(nn.JoinTable(2)) -- batchSize x seqLen*dim
                 :add(recViewer) -- batchSize x seqLen x dim
                 :add(nn.ConcatTable()
                        :add(nn.Sequential()
                            -- maybe no pad, since unreliable?
             	            :add(cudnn.TemporalConvolution(opt.rnn_size, opt.nfilters, 3, 1, 0))
             		        :add(nn.ReLU())
             		        :add(nn.Max(2)))
             	         :add(nn.Sequential()
                            -- maybe no pad, since unreliable?
             	            :add(cudnn.TemporalConvolution(opt.rnn_size, opt.nfilters, 5, 1, 0))
             		        :add(nn.ReLU())
             		        :add(nn.Max(2))))
                 :add(nn.JoinTable(2)) -- batchSize x 2*numFilters

    local nWindowFeats = 2*opt.nfilters

    if not opt.discrec then
        -- maybe want some mlp stuff first?
        mod:add(nn.Linear(nWindowFeats, opt.rnn_size*opt.nrecpreds)) -- batchSize x numPreds*rnnSize
    else
        mod:add(nn.Linear(nWindowFeats, opt.recembsize*opt.nrecpreds))
        mod:add(nn.ReLU())
        mod:add(nn.View(-1, opt.nrecpreds, opt.recembsize)) -- batchSize x numPreds x srcEmbSize

        assert(not partitionFeats or opt.recembsize % nDiscFeatures == 0)
        local featEmbDim = partitionFeats and opt.recembsize/nDiscFeatures or opt.recembsize

        local cat = nn.ConcatTable()
        for i = 1, nDiscFeatures do
            local featPredictor = nn.Sequential()

            if partitionFeats then
                assert(opt.recembsize % nDiscFeatures == 0)
                featPredictor:add(nn.Narrow(2, (i-1)*featEmbDim+1, featEmbDim))
            end

            featPredictor:add(nn.Bottle( nn.Sequential()
                                           :add(nn.Linear(featEmbDim, outVocabSize[i]))
                                           :add(nn.LogSoftMax()) ))
            cat:add(featPredictor)
        end
        mod:add(cat) -- nDiscFeatures length table of batchSize x numPreds x outVocabSize tensors
        mod:add(nn.JoinTable(3)) -- batchSize x numPreds x sum[outVocabSizes]
    end

    return mod, recViewer
end


local function buildDecoder(opt, dicts, verbose, tripV)
  local inputNetwork, inputSize = buildInputNetwork(opt, dicts, opt.pre_word_vecs_dec, opt.fix_word_vecs_dec)

  local generator

  if #dicts.features > 0 then
    generator = onmt.FeaturesGenerator.new(opt.rnn_size, dicts.words:size(), dicts.features)
  elseif opt.copy_generate then
    if opt.poe then
        generator = onmt.CopyPOEGenerator.new(opt.rnn_size, dicts.words:size(), opt.tanh_query, opt.double_output)
    else
        generator = onmt.CopyGenerator2.new(opt.rnn_size, dicts.words:size(), opt.tanh_query, opt.double_output)
    end
  else
    generator = onmt.Generator.new(opt.rnn_size, dicts.words:size())
  end

  if opt.input_feed == 1 then
    if verbose then
      print(" * using input feeding")
    end
    inputSize = inputSize + opt.rnn_size
  end

  local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual, opt.double_output)

  if opt.copy_generate then
      if opt.discrec or opt.recdist > 0 then
          local rec, recViewer = buildRec(opt, tripV)
          return onmt.ConvRecDecoder.new(inputNetwork, rnn, generator, opt.input_feed == 1,
                       opt.double_output, rec, recViewer, opt.rho)
      else
          return onmt.Decoder2.new(inputNetwork, rnn, generator, opt.input_feed == 1, opt.double_output)
      end
  else
    return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1,
      opt.copy_generate, opt.tanh_query)
  end
end

--[[ This is useful when training from a model in parallel mode: each thread must own its model. ]]
local function clonePretrained(model)
  local clone = {}

  for k, v in pairs(model) do
    if k == 'modules' then
      clone.modules = {}
      for i = 1, #v do
        table.insert(clone.modules, onmt.utils.Tensor.deepClone(v[i]))
      end
    else
      clone[k] = v
    end
  end

  return clone
end

local function loadEncoder(pretrained, clone)
  local brnn = #pretrained.modules == 2

  if clone then
    pretrained = clonePretrained(pretrained)
  end

  if brnn then
    return onmt.BiEncoder.load(pretrained)
  else
    return onmt.Encoder.load(pretrained)
  end
end

local function loadDecoder(pretrained, clone)
  if clone then
    pretrained = clonePretrained(pretrained)
  end

  return onmt.Decoder.load(pretrained)
end

return {
  buildEncoder = buildEncoder,
  buildDecoder = buildDecoder,
  loadEncoder = loadEncoder,
  loadDecoder = loadDecoder
}
