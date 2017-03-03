require('onmt.init')

local tds = require('tds')
local path = require('pl.path')
local cjson = require('cjson')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("preprocess.lua")
cmd:text("")
cmd:text("**Preprocess Options**")
cmd:text("")
cmd:text("")
cmd:option('-config', '', [[Read options from this file]])

cmd:option('-json_src', '', [[Path to the training source data]])

cmd:option('-save_data', '', [[Output file for the prepared data]])

cmd:option('-src_vocab_size', 50000, [[Size of the source vocabulary]])
cmd:option('-tgt_vocab_size', 50000, [[Size of the target vocabulary]])
cmd:option('-src_vocab', '', [[Path to an existing source vocabulary]])
cmd:option('-tgt_vocab', '', [[Path to an existing target vocabulary]])
cmd:option('-features_vocabs_prefix', '', [[Path prefix to existing features vocabularies]])

cmd:option('-src_seq_length', 50, [[Maximum source sequence length]])
cmd:option('-tgt_seq_length', 50, [[Maximum target sequence length]])
cmd:option('-shuffle', 1, [[Shuffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:option('-players_per_team', 13, [[]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])

local opt = cmd:parse(arg)

local function hasFeatures(filename)
  local reader = onmt.utils.FileReader.new(filename)
  local _, _, numFeatures = onmt.utils.Features.extract(reader:next())
  reader:close()
  return numFeatures > 0
end

local bs_keys = {"PLAYER_NAME", "START_POSITION", "MIN", "PTS", "FGM", "FGA",
     "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
     "DREB", "REB", "AST", "TO", "STL", "BLK", "PF", "FIRST_NAME",
     "SECOND_NAME"}

local ls_keys = {"PTS_QTR1", "PTS_QTR2", "PTS_QTR3", "PTS_QTR4", "PTS",
   "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "WINS", "LOSSES",
   "CITY", "NAME"}

-- this will make vocab for every word in summary or in a table cell or header i think
local function makeVocabulary(jsondat, size)
  local wordVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                         onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  local featuresVocabs = {}

  for i = 1, #jsondat do
      local game = jsondat[i]
      -- add all the words in the summary
      for j = 1, #game.summary do
          wordVocab:add(game.summary[j])
      end
      -- add all the box score poop
      for t = 1, #bs_keys do
          local k = bs_keys[t]
          local tbl = game.box_score[k]
      --for k, tbl in pairs(game.box_score) do
          for idx, val in pairs(tbl) do
              wordVocab:add(val)
          end
      end

      -- add all the linescore stuff
      for t = 1, #ls_keys do
          local k = ls_keys[t]
          local v = game.home_line[k]
      --for k, v in pairs(game.home_line) do
          wordVocab:add(v)
          wordVocab:add(game.vis_line[k])
      end
    --   -- also add cities and names
    --   wordVocab:add(game.home_name)
    --   wordVocab:add(game.home_city)
    --   wordVocab:add(game.vis_name)
    --   wordVocab:add(game.vis_city)
  end

  local originalSize = wordVocab:size()
  wordVocab = wordVocab:prune(size)
  print('Created dictionary of size ' .. wordVocab:size() .. ' (pruned from ' .. originalSize .. ')')

  return wordVocab, featuresVocabs
end

local function initVocabulary(name, jsondat, vocabFile, vocabSize, featuresVocabsFiles)
  local wordVocab
  local featuresVocabs = {}

  if vocabFile:len() > 0 then
    -- If given, load existing word dictionary.
    print('Reading ' .. name .. ' vocabulary from \'' .. vocabFile .. '\'...')
    wordVocab = onmt.utils.Dict.new()
    wordVocab:loadFile(vocabFile)
    print('Loaded ' .. wordVocab:size() .. ' ' .. name .. ' words')
  end

  if featuresVocabsFiles:len() > 0 then
    -- If given, discover existing features dictionaries.
    local j = 1

    while true do
      local file = featuresVocabsFiles .. '.' .. name .. '_feature_' .. j .. '.dict'

      if not path.exists(file) then
        break
      end

      print('Reading ' .. name .. ' feature ' .. j .. ' vocabulary from \'' .. file .. '\'...')
      featuresVocabs[j] = onmt.utils.Dict.new()
      featuresVocabs[j]:loadFile(file)
      print('Loaded ' .. featuresVocabs[j]:size() .. ' labels')

      j = j + 1
    end
  end

  if wordVocab == nil or (#featuresVocabs == 0 and hasFeatures(dataFile)) then
    -- If a dictionary is still missing, generate it.
    print('Building ' .. name  .. ' vocabulary...')
    local genWordVocab, genFeaturesVocabs = makeVocabulary(jsondat, vocabSize)

    if wordVocab == nil then
      wordVocab = genWordVocab
    end
    if #featuresVocabs == 0 then
      featuresVocabs = genFeaturesVocabs
    end
  end

  print('')

  return {
    words = wordVocab,
    features = featuresVocabs
  }
end

local function saveVocabulary(name, vocab, file)
  print('Saving ' .. name .. ' vocabulary to \'' .. file .. '\'...')
  vocab:writeFile(file)
end

local function saveFeaturesVocabularies(name, vocabs, prefix)
  for j = 1, #vocabs do
    local file = prefix .. '.' .. name .. '_feature_' .. j .. '.dict'
    print('Saving ' .. name .. ' feature ' .. j .. ' vocabulary to \'' .. file .. '\'...')
    vocabs[j]:writeFile(file)
  end
end

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i, v in pairs(vec) do
    t[i] = v
  end
  return t
end

local function get_player_idxs(game, max_per_team)
    local home_players, vis_players = {}, {}
    -- count total number of players
    local nplayers = 0
    for k,v in pairs(game.box_score['PTS']) do
        nplayers = nplayers + 1
    end

    local num_home, num_vis = 0, 0
    for i = 1, nplayers do
        local player_city = game.box_score.TEAM_CITY[tostring(i-1)]
        if player_city == game.home_city then
            if #home_players < max_per_team then
                table.insert(home_players, tostring(i-1))
                num_home = num_home + 1
            end
        else
            if #vis_players < max_per_team then
                table.insert(vis_players, tostring(i-1))
                num_vis = num_vis + 1
            end
        end
    end
    --print("adding", num_home, num_vis, "players")
    return home_players, vis_players
end

local function makeData(jsondat, srcDicts, tgtDicts, shuffle)
  -- i guess make an input sequence for every player and every team
  -- can add city as a timestep, but maybe just add it as a feature

  local players_per_team = opt.players_per_team
  -- i guess make a src for each row
  local srcs = {}
  for i = 1, 2*players_per_team+2 do -- 2 teams
      table.insert(srcs, tds.Vec())
  end
  local srcFeatures = tds.Vec()

  local tgt = tds.Vec()
  local tgtFeatures = tds.Vec()

  local sizes = tds.Vec() -- will be target sizes...

  local count = 0
  local ignored = 0

  -- local srcReader = onmt.utils.FileReader.new(srcFile)
  -- local tgtReader = onmt.utils.FileReader.new(tgtFile)


  for i = 1, #jsondat do
    -- local srcTokens = srcReader:next()
    -- local tgtTokens = tgtReader:next()
    local game = jsondat[i]
    -- get player_idxs for each team, since there're not always 13 of each
    local home_players, vis_players = get_player_idxs(game, players_per_team)

    local tgtTokens = game.summary

    -- if srcTokens == nil or tgtTokens == nil then
    --   if srcTokens == nil and tgtTokens ~= nil or srcTokens ~= nil and tgtTokens == nil then
    --     print('WARNING: source and target do not have the same number of sentences')
    --   end
    --   break
    -- end

    --if #srcTokens > 0 and #srcTokens <= opt.src_seq_length
    if #tgtTokens > 0 and #tgtTokens <= opt.tgt_seq_length then
      --local srcWords, srcFeats = onmt.utils.Features.extract(srcTokens)
      --local tgtWords, tgtFeats = onmt.utils.Features.extract(tgtTokens)
      local tgtWords = tgtTokens

      for ii, player_list in ipairs({home_players, vis_players}) do
          for j = 1, players_per_team do
              local src_j = {}
              local player_key = player_list[j] -- can be nil if not enough
              for k, key in ipairs(bs_keys) do
                  local val = game.box_score[key][player_key]
                  assert(val or (not player_key))
                  table.insert(src_j, val or "N/A")
              end
              local idxs = srcDicts.words:convertToIdx(src_j, onmt.Constants.UNK_WORD)
              assert(idxs:dim() > 0)
              srcs[(ii-1)*players_per_team+j]:insert(idxs)
          end
      end

      -- make line scores the same size as box scores by pre-padding
      local home_src, vis_src = {}, {}
      for j = 1, (#bs_keys - #ls_keys) do
          table.insert(home_src, onmt.Constants.PAD_WORD)
          table.insert(vis_src, onmt.Constants.PAD_WORD)
      end
    --   -- add city and team names
    --   table.insert(home_src, game.home_city)
    --   table.insert(home_src, game.home_name)
    --   table.insert(vis_src, game.vis_city)
    --   table.insert(vis_src, game.vis_name)
      -- add rest of the stuff
      for k, key in ipairs(ls_keys) do
          --print(k, key, game.home_line[key])
          table.insert(home_src, game.home_line[key])
          table.insert(vis_src, game.vis_line[key])
      end

      assert(#home_src == srcs[1][1]:size(1))
      assert(#vis_src == srcs[1][1]:size(1))
      local idxs = srcDicts.words:convertToIdx(home_src, onmt.Constants.UNK_WORD)
      assert(idxs:dim() > 0)
      srcs[2*players_per_team+1]:insert(idxs)
      idxs = srcDicts.words:convertToIdx(vis_src, onmt.Constants.UNK_WORD)
      assert(idxs:dim() > 0)
      srcs[2*players_per_team+2]:insert(idxs)


      --src:insert(srcDicts.words:convertToIdx(srcWords, onmt.Constants.UNK_WORD))
      tgt:insert(tgtDicts.words:convertToIdx(tgtWords,
                                             onmt.Constants.UNK_WORD,
                                             onmt.Constants.BOS_WORD,
                                             onmt.Constants.EOS_WORD))

      if #srcDicts.features > 0 then
        srcFeatures:insert(onmt.utils.Features.generateSource(srcDicts.features, srcFeats, true))
      end
      if #tgtDicts.features > 0 then
        tgtFeatures:insert(onmt.utils.Features.generateTarget(tgtDicts.features, tgtFeats, true))
      end

      sizes:insert(#tgtWords)
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % opt.report_every == 0 then
      print('... ' .. count .. ' sentences prepared')
    end
  end

  --srcReader:close()
  --tgtReader:close()

  local function reorderData(perm)
    tgt = onmt.utils.Table.reorder(tgt, perm, true)

    for j = 1, #srcs do
        srcs[j] = onmt.utils.Table.reorder(srcs[j], perm, true)
    end

    if #srcDicts.features > 0 then
      srcFeatures = onmt.utils.Table.reorder(srcFeatures, perm, true)
    end
    if #tgtDicts.features > 0 then
      tgtFeatures = onmt.utils.Table.reorder(tgtFeatures, perm, true)
    end
  end

  --if opt.shuffle == 1 then
  if shuffle then
    print('... shuffling sentences')
    local perm = torch.randperm(#tgt)
    --print(perm)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  if shuffle then
      print('... sorting sentences by size')
      local _, perm = torch.sort(vecToTensor(sizes))
      reorderData(perm)
  end

  print('Prepared ' .. #tgt .. ' sentences (' .. ignored
          .. ' ignored due to source length > ' .. opt.src_seq_length
          .. ' or target length > ' .. opt.tgt_seq_length .. ')')

  local srcData = {
    words = srcs,
    features = srcFeatures
  }

  local tgtData = {
    words = tgt,
    features = tgtFeatures
  }

  return srcData, tgtData
end

local function main()
  local requiredOptions = {
    "json_src",
    "save_data"
  }

  onmt.utils.Opt.init(opt, requiredOptions)

  local f = io.open(opt.json_src)
  local jsondat = cjson.decode(f:read("*all"))
  f:close()

  local data = {}

  data.dicts = {}
  data.dicts.src = initVocabulary('source', jsondat.train, opt.src_vocab,
                                  opt.src_vocab_size, opt.features_vocabs_prefix)
  data.dicts.tgt = data.dicts.src
  -- data.dicts.tgt = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
  --                                 opt.tgt_vocab_size, opt.features_vocabs_prefix)

  print('Preparing training data...')
  data.train = {}
  data.train.src, data.train.tgt = makeData(jsondat.train,
                                            data.dicts.src, data.dicts.tgt, true)
  print('')

  print('Preparing validation data...')
  data.valid = {}
  data.valid.src, data.valid.tgt = makeData(jsondat.valid,
                                            data.dicts.src, data.dicts.tgt, false)
  print('')

  if opt.src_vocab:len() == 0 then
    saveVocabulary('source', data.dicts.src.words, opt.save_data .. '.src.dict')
  end

  if opt.tgt_vocab:len() == 0 then
    saveVocabulary('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
  end

  if opt.features_vocabs_prefix:len() == 0 then
    saveFeaturesVocabularies('source', data.dicts.src.features, opt.save_data)
    saveFeaturesVocabularies('target', data.dicts.tgt.features, opt.save_data)
  end

  print('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)

end

main()
