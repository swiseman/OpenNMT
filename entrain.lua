require('onmt.init')

local path = require('pl.path')
require('tds')
local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data', '', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-layers', 2, [[Number of layers in the RNN encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of RNN hidden states]])
cmd:option('-rnn_type', 'LSTM', [[Type of RNN cell: LSTM, GRU]])
cmd:option('-word_vec_size', 0, [[Common word embedding size. If set, this overrides -src_word_vec_size and -tgt_word_vec_size.]])
cmd:option('-src_word_vec_size', '500', [[Comma-separated list of source embedding sizes: word[,feat1,feat2,...].]])
cmd:option('-tgt_word_vec_size', '500', [[Comma-separated list of target embedding sizes: word[,feat1,feat2,...].]])
cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
cmd:option('-feat_vec_exponent', 0.7, [[When features embedding sizes are not set and using -feat_merge concat, their dimension will be set to N^exponent where N is the number of values the feature takes.]])
cmd:option('-feat_vec_size', 20, [[When features embedding sizes are not set and using -feat_merge sum, this is the common embedding size of the features]])
cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
cmd:option('-brnn', false, [[Use a bidirectional encoder]])
cmd:option('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])
cmd:option('-margin', 1, [[]])
cmd:option('-num_sample_inputs', 5000, [[]])
cmd:option('-max_back', 2, [[]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-end_epoch', 13, [[The final epoch of the training]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-start_iteration', 1, [[If loading from a checkpoint, the iteration from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings are: sgd = 1,
                                adagrad = 0.1, adadelta = 1, adam = 0.0002]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-learning_rate_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 10000, [[Start decay after this epoch]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
onmt.utils.Cuda.declareOpts(cmd)
cmd:option('-async_parallel', false, [[Use asynchronous parallelism training.]])
cmd:option('-async_parallel_minbatch', 1000, [[For async parallel computing, minimal number of batches before being parallel.]])
cmd:option('-no_nccl', false, [[Disable usage of nccl in parallel mode.]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

-- bookkeeping
cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-json_log', false, [[Outputs logs in JSON format.]])

onmt.utils.Logger.declareOpts(cmd)
onmt.utils.Profiler.declareOpts(cmd)

local opt = cmd:parse(arg)
opt.en = true

local function initParams(model, verbose)
    local numParams = 0
    local params = {}
    local gradParams = {}

    if verbose then
        print('Initializing parameters...')
    end

    -- we assume all the sharing has already been done,
    -- so we just make a big container to flatten everything
    local everything = nn.Sequential()
    for k, mod in pairs(model) do
        everything:add(mod)
    end

    local p, gp = everything:getParameters()

    if opt.train_from:len() == 0 then
        p:uniform(-opt.param_init, opt.param_init)
        -- do module specific init; wordembeddings will happen multiple times,
        -- but who cares
        for k, mod in pairs(model) do
            mod:apply(function (m)
                if m.postParametersInitialization then
                    m:postParametersInitialization()
                end
            end)
        end
    else
        print("copying loaded params...")
        local checkpoint = torch.load(opt.train_from)
        p:copy(checkpoint.flatParams[1])
    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)
    if verbose then
        print(" * number of parameters: " .. numParams)
    end
    return params, gradParams
end


local function buildCriterion(vocabSize, features)
  local criterion = nn.MarginRankingCriterion(opt.margin)
  criterion.sizeAverage = false -- should maybe not do this...
  return criterion
end


function allTraining(model)
    for _, mod in pairs(model) do
        if mod.training then
            mod:training()
        end
    end
end

function allEvaluate(model)
    for _, mod in pairs(model) do
        if mod.evaluate then
            mod:evaluate()
        end
    end
end


local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  model.encoder:evaluate()
  model.decoder:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local encoderStates, context = model.encoder:forward(batch)
    loss = loss + model.decoder:computeLoss(batch, encoderStates, context, criterion)
    total = total + batch.targetNonZeros
  end

  model.encoder:training()
  model.decoder:training()

  return loss/total --math.exp(loss / total)
end

local function trainModel(model, trainData, validData, dataset, info)
  local params, gradParams = initParams(model)
  local criterion
  allTraining(model)

  -- define criterion of each GPU
  criterion = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words:size(),
                                                          dataset.dicts.tgt.features))

  -- optimize memory of the first clone
  if not opt.disable_mem_optimization then
    local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
    batch.totalSize = batch.size
    onmt.utils.Memory.optimize(model, criterion, batch, verbose)
  end

  local optim = onmt.train.Optim.new({
    method = opt.optim,
    numModels = 1,
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    startDecayAt = opt.start_decay_at,
    optimStates = opt.optim_states
  })

  local checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset.dicts)

  local function trainEpoch(epoch, lastValidPpl, doProfile)
      local epochState
      local batchOrder

      local epochProfiler = onmt.utils.Profiler.new(doProfile)

      local startI = opt.start_iteration

      local numIterations = trainData:batchCount()

      if startI > 1 and info ~= nil then
         epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
         batchOrder = info.batchOrder
      else
         epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, nil, true)
         -- Shuffle mini batch order.
         batchOrder = torch.randperm(trainData:batchCount())
      end

     --opt.start_iteration = 1

      local iter = 1
      for i = startI, trainData:batchCount() do
          local batchIdx = epoch <= opt.curriculum and i or batchOrder[i]
          local batch = trainData:getBatch(batchIdx)
          batch.totalSize = batch.size
          if opt.curriculum > 0 then
              batch.targetLength = math.min(batch.targetLength, epoch)
          end
          onmt.utils.Cuda.convert(batch)

          optim:zeroGrad(gradParams)
          --_G.profiler:start("encoder.fwd")
          local encStates, context = model.encoder:forward(batch)
          --_G.profiler:stop("encoder.fwd"):start("decoder.fwd")
          local cumSums = model.decoder:forward(batch, encStates, context)
          --_G.profiler:stop("decoder.fwd"):start("decoder.bwd")
          local encGradStatesOut, gradContext, loss = model.decoder:backward(batch, cumSums, criterion)
          --_G.profiler:stop("decoder.bwd"):start("encoder.bwd")
          model.encoder:backward(batch, encGradStatesOut, gradContext)
          --_G.profiler:stop("encoder.bwd")

          -- Update the parameters.
          optim:prepareGrad(gradParams, opt.max_grad_norm)
          optim:updateParams(params, gradParams)
          epochState:update(batch, loss)

          if iter % opt.report_every == 0 then
              epochState:log(iter, opt.json_log)
          end
          if opt.save_every > 0 and iter % opt.save_every == 0 then
             checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
          end
          iter = iter + 1
      end
      return epochState, epochProfiler:dump()
  end

  local validPpl = 0

  if not opt.json_log then
    _G.logger:info('Start training...')
  end

  for epoch = opt.start_epoch, opt.end_epoch do
    if not opt.json_log then
      _G.logger:info('')
    end

    local globalProfiler = onmt.utils.Profiler.new(opt.profiler)

    globalProfiler:start("train")
    local epochState, epochProfile = trainEpoch(epoch, validPpl, opt.profiler)
    globalProfiler:add(epochProfile)
    epochState:log(epochState.numIterations)
    globalProfiler:stop("train")

    globalProfiler:start("valid")
    validPpl = eval(model, criterion, validData)
    globalProfiler:stop("valid")

    if not opt.json_log then
      if opt.profiler then _G.logger:info('profile: %s', globalProfiler:log()) end
      _G.logger:info('Validation Loss: %.3f', validPpl)
    end

    if opt.optim == 'sgd' then
        if opt.decay_update2 then
            optim:updateLearningRate2(validPpl, epoch)
        else
            optim:updateLearningRate(validPpl, epoch)
        end
    end

    --checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
  end
end


local function main()
  local requiredOptions = {
    "data",
    "save_model"
  }

-- need to allow stepsBack to vary again, don't break after single timestep,
-- switch tanh to relu, :float() instead of :double()
--  assert(false)

  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new(false)

  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint = {}

  -- Create the data loader class.
  if not opt.json_log then
    _G.logger:info('Loading data from \'' .. opt.data .. '\'...')
  end

  local dataset = torch.load(opt.data, 'binary', false)

  local trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt, true)
  local validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt, true)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  if not opt.json_log then
    _G.logger:info(' * vocabulary size: source = %d; target = %d',
                   dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size())
    _G.logger:info(' * additional features: source = %d; target = %d',
                   #dataset.dicts.src.features, #dataset.dicts.tgt.features)
    _G.logger:info(' * maximum sequence length: source = %d; target = %d',
                   trainData.maxSourceLength, trainData.maxTargetLength)
    _G.logger:info(' * number of training sentences: %d', #trainData.src)
    _G.logger:info(' * maximum batch size: %d', opt.max_batch_size)
  else
    local metadata = {
      options = opt,
      vocabSize = {
        source = dataset.dicts.src.words:size(),
        target = dataset.dicts.tgt.words:size()
      },
      additionalFeatures = {
        source = #dataset.dicts.src.features,
        target = #dataset.dicts.tgt.features
      },
      sequenceLength = {
        source = trainData.maxSourceLength,
        target = trainData.maxTargetLength
      },
      trainingSentences = #trainData.src
    }

    onmt.utils.Log.logJson(metadata)
  end

  if not opt.json_log then
    _G.logger:info('Building model...')
  end

  local model = {}
  model.encoder = onmt.Models.buildEncoder(opt, dataset.dicts.src)
  model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose)

  for _, mod in pairs(model) do
    onmt.utils.Cuda.convert(mod)
  end

  trainModel(model, trainData, validData, dataset, checkpoint.info)

  _G.logger:shutDown()
end

main()
