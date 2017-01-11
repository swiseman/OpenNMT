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

cmd:option('-layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
cmd:option('-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                      then the embedding dimension will be set to N^exponent]])
cmd:option('-feat_vec_size', 20, [[When using sum, the common embedding size of the features]])
cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
cmd:option('-brnn', false, [[Use a bidirectional encoder]])
cmd:option('-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
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
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
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
cmd:option('-tie_encoder_rnns', false, [[Tie all encoder rnns]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', 0, [[1-based identifier of the GPU to use. CPU is used when the option is < 1]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

-- bookkeeping
cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-json_log', false, [[Outputs logs in JSON format.]])

local opt = cmd:parse(arg)

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
    else
        assert(false)
    end

    -- do module specific init; wordembeddings will happen multiple times,
    -- but who cares
    for k, mod in pairs(model) do
        mod:apply(function (m)
            if m.postParametersInitialization then
                m:postParametersInitialization()
            end
        end)
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
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  addNllCriterion(vocabSize)

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  return criterion
end

local function allTraining(model)
    for _, mod in pairs(model) do
        if mod.training then
            mod:training()
        end
    end
end

local function allEvaluate(model)
    for _, mod in pairs(model) do
        if mod.evaluate then
            mod:evaluate()
        end
    end
end

-- gets encodings for all rows
function allEncForward(model, batch)
    --for k,v in pairs(model) do print(k) end
    local allEncStates, allCtxs = {}, {}
    for j = 1, batch.sourceInput:size(1) do
        batch:setInputRow(j)
        local encStates, context = model["encoder" .. j]:forward(batch)
        table.insert(allEncStates, encStates)
        table.insert(allCtxs, context)
    end
    batch:setInputRow(nil) -- for sanity
    local aggEncStates, catCtx = model.aggregator:forward(allEncStates, allCtxs)
    return aggEncStates, catCtx
end

-- goes backward over all encoders
function allEncBackward(model, batch, encGradStatesOut, gradContext)
    local allEncGradOuts, gradCtxs = model.aggregator:backward(encGradStatesOut, gradContext, opt.input_feed)
    for j = 1, batch.sourceInput:size(1) do
        batch:setInputRow(j)
        model["encoder" .. j]:backward(batch, allEncGradOuts[j], gradCtxs[j])
    end
    batch:setInputRow(nil)
end

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  -- model.encoder:evaluate()
  -- model.decoder:evaluate()
  allEvaluate(model)

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    --local encoderStates, context = model.encoder:forward(batch)
    local allEncStates, allCtxs = allEncForward(model, batch)
    local aggEncStates, catCtx = model.aggregator:forward(allEncStates, allCtxs)
    --loss = loss + model.decoder:computeLoss(batch, encoderStates, context, criterion)
    loss = loss + model.decoder:computeLoss(batch, aggEncStates, catCtx, criterion)
    total = total + batch.targetNonZeros
  end

  -- model.encoder:training()
  -- model.decoder:training()
  allTraining(model)

  return math.exp(loss / total)
end

local function trainModel(model, trainData, validData, dataset, info)
    local criterion
    local verbose = true
    local params, gradParams = initParams(model, verbose)
    allTraining(model)
    -- for _, mod in pairs(model) do
    --     mod:training()
    -- end

    -- define criterion of each GPU
    criterion = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words:size(),
                                                          dataset.dicts.tgt.features))

    -- optimize memory of the first clone
    if not opt.disable_mem_optimization then
        local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
        batch.totalSize = batch.size
        onmt.utils.Memory.boxOptimize(model, trainData.nSourceRows, criterion, batch, verbose)
    end

    local optim = onmt.train.Optim.new({
        method = opt.optim,
        numModels = 1, -- we flattened everything
        learningRate = opt.learning_rate,
        learningRateDecay = opt.learning_rate_decay,
        startDecayAt = opt.start_decay_at,
        optimStates = opt.optim_states
    })

    local checkpoint = onmt.train.Checkpoint.new(opt, model, optim, dataset)

    local function trainEpoch(epoch, lastValidPpl)
        local epochState
        local batchOrder
        local startI = opt.start_iteration

        local numIterations = trainData:batchCount()

        if startI > 1 and info ~= nil then
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
            batchOrder = info.batchOrder
        else
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
            -- Shuffle mini batch order.
            batchOrder = torch.randperm(trainData:batchCount())
        end

        --opt.start_iteration = 1

        local iter = 1
        for i = startI, trainData:batchCount() do
            local batchIdx = epoch <= opt.curriculum and i or batchOrder[i]
            local batch =  trainData:getBatch(batchIdx)
            batch.totalSize = batch.size -- fuck off
            onmt.utils.Cuda.convert(batch)

            optim:zeroGrad(gradParams)
            local aggEncStates, catCtx = allEncForward(model, batch)
            local ctxLen = catCtx:size(2)
            local decOutputs = model.decoder:forward(batch, aggEncStates, catCtx)
            local encGradStatesOut, gradContext, loss = model.decoder:backward(batch, decOutputs, criterion, ctxLen)
            allEncBackward(model, batch, encGradStatesOut, gradContext)

            -- Update the parameters.
            optim:prepareGrad(gradParams[1], opt.max_grad_norm)
            optim:updateParams(params[1], gradParams[1])
            epochState:update(batch, loss)

            if iter % opt.report_every == 0 then
                epochState:log(iter, opt.json_log)
            end
            if opt.save_every > 0 and iter % opt.save_every == 0 then
                checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
            end
            iter = iter + 1
        end
        return epochState
    end -- end local function trainEpoch

    local validPpl = 0

    if not opt.json_log then
        print('Start training...')
    end

    for epoch = opt.start_epoch, opt.epochs do
        if not opt.json_log then
            print('')
        end

        local epochState = trainEpoch(epoch, validPpl)
        validPpl = eval(model, criterion, validData)
        if not opt.json_log then
            print('Validation perplexity: ' .. validPpl)
        end

        if opt.optim == 'sgd' then
            optim:updateLearningRate(validPpl, epoch)
        end

        checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
    end

end -- end local function trainModel

-- ties word embeddings of all encoders to decoder embeddings, and optionally
-- ties encoder rnns (to each other)
local function makeVariouslyTiedEncoder(opt, srcDict, decoder, encSharee)
    local decWordEmb = decoder.inputNet
    assert(torch.type(decWordEmb) == "onmt.WordEmbedding")
    local enc = onmt.Models.buildEncoder(opt, srcDict)
    onmt.utils.Cuda.convert(enc) -- share from gpu
    if opt.brnn then -- reshare both word embeddings
        assert(torch.type(enc.fwd.inputNet) == "onmt.WordEmbedding")
        enc.fwd.inputNet.net:share(decWordEmb.net, 'weight', 'gradWeight')
        assert(torch.type(enc.bwd.inputNet) == "onmt.WordEmbedding")
        enc.bwd.inputNet.net:share(decWordEmb.net, 'weight', 'gradWeight')
        if encSharee then -- share recurrent params with this encoder
            enc.bwd.rnn.net:share(encSharee.fwd.rnn.net, 'weight', 'gradWeight', 'bias', 'gradBias')
            enc.fwd.rnn.net:share(encSharee.fwd.rnn.net, 'weight', 'gradWeight', 'bias', 'gradBias')
        else -- just share fwd and bwd within this rnn
            enc.bwd.rnn.net:share(enc.fwd.rnn.net, 'weight', 'gradWeight', 'bias', 'gradBias')
        end
    else
        assert(torch.type(enc.inputNet) == "onmt.WordEmbedding")
        enc.inputNet.net:share(decWordEmb.net, 'weight', 'gradWeight')
        if encSharee then
            enc.rnn.net:share(encSharee.rnn.net, 'weight', 'gradWeight', 'bias', 'gradBias')
        end
    end
    return enc
end

local function main()
  local requiredOptions = {
    "data",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  local checkpoint = {}

  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')

    if not opt.json_log then
      print('Loading checkpoint \'' .. opt.train_from .. '\'...')
    end

    checkpoint = torch.load(opt.train_from)

    opt.layers = checkpoint.options.layers
    opt.rnn_size = checkpoint.options.rnn_size
    opt.brnn = checkpoint.options.brnn
    opt.brnn_merge = checkpoint.options.brnn_merge
    opt.input_feed = checkpoint.options.input_feed

    -- Resume training from checkpoint
    if opt.train_from:len() > 0 and opt.continue then
      opt.optim = checkpoint.options.optim
      opt.learning_rate_decay = checkpoint.options.learning_rate_decay
      opt.start_decay_at = checkpoint.options.start_decay_at
      opt.epochs = checkpoint.options.epochs
      opt.curriculum = checkpoint.options.curriculum

      opt.learning_rate = checkpoint.info.learningRate
      opt.optim_states = checkpoint.info.optimStates
      opt.start_epoch = checkpoint.info.epoch
      opt.start_iteration = checkpoint.info.iteration

      if not opt.json_log then
        print('Resuming training from epoch ' .. opt.start_epoch
                .. ' at iteration ' .. opt.start_iteration .. '...')
      end
    end
  end

  -- Create the data loader class.
  if not opt.json_log then
    print('Loading data from \'' .. opt.data .. '\'...')
  end

  local dataset = torch.load(opt.data, 'binary', false)

  local trainData = onmt.data.BoxDataset.new(dataset.train.src, dataset.train.tgt)
  local validData = onmt.data.BoxDataset.new(dataset.valid.src, dataset.valid.tgt)

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  if not opt.json_log then
    print(string.format(' * vocabulary size: source = %d; target = %d',
                        dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size()))
    print(string.format(' * additional features: source = %d; target = %d',
                        #dataset.dicts.src.features, #dataset.dicts.tgt.features))
    print(string.format(' * maximum sequence length: source = %d; target = %d',
                        trainData.maxSourceLength, trainData.maxTargetLength))
    print("nSourceRows", trainData.nSourceRows)
    print(string.format(' * number of training instances: %d', #trainData.tgt))
    print(string.format(' * maximum batch size: %d', opt.max_batch_size))
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
      trainingSentences = #trainData.tgt
    }

    onmt.utils.Log.logJson(metadata)
  end

  if not opt.json_log then
    print('Building model...')
  end

    local model = {}

    local verbose = true
    if checkpoint.models then
        assert(false) -- this is gonna be annoying
        for i = 1, trainData.nSourceRows do
            model["encoder" .. i] = onmt.Models.loadEncoder(checkpoint.models["encoder" .. i], false)
        end
        model.decoder = onmt.Models.loadDecoder(checkpoint.models.decoder, false)
    else
        -- make decoder first
        model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose)
        -- send to gpu immediately to make cloning things simpler
        onmt.utils.Cuda.convert(model.decoder)

        for i = 1, trainData.nSourceRows do
            --model["encoder" .. i] = onmt.Models.buildEncoder(opt, dataset.dicts.src)
            local sharee = opt.tie_encoder_rnns and model.encoder1
            model["encoder" .. i] = makeVariouslyTiedEncoder(opt, dataset.dicts.src, model.decoder, sharee)
            -- already on gpu
        end

        model.aggregator = onmt.Aggregator(trainData.nSourceRows, opt.rnn_size, opt.rnn_size)
        onmt.utils.Cuda.convert(model.aggregator)

    end

    -- for _, mod in pairs(model) do
    --     onmt.utils.Cuda.convert(mod)
    -- end

    trainModel(model, trainData, validData, dataset, checkpoint.info)
end

main()
