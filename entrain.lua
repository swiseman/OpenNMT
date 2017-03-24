require('onmt.init')

local path = require('pl.path')
require('tds')
local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data', '', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-just_eval', false, [[]])


cmd:text("")
cmd:text("**Model options**")
cmd:text("")
cmd:option('-margin', 1, [[]])
cmd:option('-num_sample_inputs', 500, [[]])
cmd:option('-max_back', 1, [[]])
cmd:option('-sample_scheme', 'cp', [[cp|batch|random]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-start_decay_at', 10000, [[Start decay after this epoch]])
cmd:option('-decay_update2', false, [[Decay less]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-seed', 3435, [[Seed for random initialization]])

-- i guess get all the shit
onmt.Model.declareOpts(cmd)
onmt.Seq2Seq.declareOpts(cmd)
onmt.train.Optim.declareOpts(cmd)
onmt.train.Trainer.declareOpts(cmd)
onmt.train.Checkpoint.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Memory.declareOpts(cmd)
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
  if model.decoder.cpModel then
      model.decoder.cpModel:evaluate()
  end

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local encoderStates, context = model.encoder:forward(batch)
    loss = loss + model.decoder:computeLoss(batch, encoderStates, context, criterion)
    total = total + batch.targetNonZeros
  end

  model.encoder:training()
  model.decoder:training()
  if model.decoder.cpModel then
      model.decoder.cpModel:training()
  end

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

  local checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset.dicts, params)

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
              batch.targetLength = math.min(batch.targetLength, 2*epoch)
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
  local bestPpl = math.huge
  local bestEpoch = -1

  if not opt.json_log then
    _G.logger:info('Start training...')
  end

  if opt.just_eval then
      onmt.train.Greedy.greedy_eval(model, validData, nil, g_tgtDict, 1, 10)
      return
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

    -- globalProfiler:start("valid")
    -- validPpl = eval(model, criterion, validData)
    -- globalProfiler:stop("valid")
    globalProfiler:start("val gbleu")
    -- negate...
    validPpl = -onmt.train.Greedy.greedy_eval(model, validData, nil, g_tgtDict, 1, 10, 120)
    globalProfiler:stop("val gbleu")

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

    if validPpl < bestPpl then
        checkpoint:deleteEpoch(bestPpl, bestEpoch)
        checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
        bestPpl = validPpl
        bestEpoch = epoch
    end
    --checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
  end
end


local function main()
  -- local requiredOptions = {
  --   "data",
  --   "save_model"
  -- }

--  onmt.utils.Opt.init(opt, requiredOptions)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new(false)

  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint = {}

  -- Create the data loader class.

  _G.logger:info('Loading data from \'' .. opt.data .. '\'...')


  local dataset = torch.load(opt.data, 'binary', false)
  dataset.dataType = dataset.dataType or 'bitext'
  g_tgtDict = dataset.dicts.tgt.words

  local trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt, true)
  local validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt, true)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size) -- maybe set this to be lower???
  --validData:setBatchSize(16)


  _G.logger:info(' * vocabulary size: source = %d; target = %d',
                   dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size())
  _G.logger:info(' * additional features: source = %d; target = %d',
                   #dataset.dicts.src.features, #dataset.dicts.tgt.features)
  _G.logger:info(' * maximum sequence length: source = %d; target = %d',
                   trainData.maxSourceLength, trainData.maxTargetLength)
  _G.logger:info(' * number of training sentences: %d', #trainData.src)
  _G.logger:info(' * maximum batch size: %d', opt.max_batch_size)

  _G.logger:info('Building model...')


  local model = {}
  model.encoder = onmt.Factory.buildWordEncoder(opt, dataset.dicts.src)
  model.decoder = onmt.NugDecoder(dataset.dicts.targ)

  for _, mod in pairs(model) do
    onmt.utils.Cuda.convert(mod)
  end

  trainModel(model, trainData, validData, dataset, checkpoint.info)

  _G.logger:shutDown()
end

main()
