--[[ Return the maxLength, sizes, and non-zero count
  of a baBoxBatch`seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0
  local sum = 0

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    max = math.max(max, len)
    sum = sum + len
    sizes[i] = len
  end
  return max, sizes, sum
end

--[[ Data management and batch creation.

Batch interface reference [size]:

  * size: number of sentences in the batch [1]
  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * sourceInputRev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * sourceInputRevFeatures: table of reversed source features sequences
  * targetLength: max length in source batch [1]
  * targetSize: lengths of each source [batch x 1]
  * targetNonZeros: number of non-ignored words in batch [1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]

--[[ A batch of sentences to translate and targets. Manages padding,
  features, and batch alignment (for efficiency).

  Used by the decoder and encoder objects.
--]]
local BoxBatch2 = torch.class('BoxBatch2')

--[[ Create a batch object.

Parameters:

  * `src` - 2D table of source batch indices
  * `srcFeatures` - 2D table of source batch features (opt)
  * `tgt` - 2D table of target batch indices
  * `tgtFeatures` - 2D table of target batch features (opt)
--]]
function BoxBatch2:__init(srcs, srcFeatures, tgt, tgtFeatures, bsLen,
    colStartIdx, nFeatures, targetMasks)
  local srcs = srcs or {}

  if tgt ~= nil then
    assert(#srcs[1] == #tgt, "source and target must have the same batch size")
  end

  self.size = #tgt

  self.sourceLength = bsLen-1 -- skipping first col...
  self.totalSourceLength = #srcs*self.sourceLength -- all rows
  assert(srcs[1][1]:size(1) == bsLen)
  local srcLen = self.sourceLength
  local vocabSize = colStartIdx+2*srcLen+1
  --self.sourceLength, self.sourceSize = getLength(src)

  --local sourceSeq = torch.IntTensor(#srcs, self.sourceLength, self.size):fill(onmt.Constants.PAD)
  -- source concatenates all rows in the table into a single column (and concatenates everything in the batch too)
  self.sourceInput = torch.IntTensor(self.size*self.totalSourceLength, nFeatures)
  --self.sourceInput = sourceSeq:clone()

  local srcLocs
  if targetMasks then
      srcLocs = findSourceLocations(srcs, self.size, vocabSize)
  end

  local maxIndices = 7 -- I'm just assuming we never get more than 8; deal with it

  if tgt ~= nil then
    self.targetLength, self.targetSize, self.targetNonZeros = getLength(tgt, 1)

    local targetSeq = torch.IntTensor(self.targetLength, self.size):fill(onmt.Constants.PAD)
    self.targetInput = targetSeq:clone()
    -- this might be too big
    self.targetOutput = targetMasks and torch.IntTensor(self.targetLength, self.size, maxIndices+1):zero() or targetSeq:clone()
  end



  local currRow = 1

  for b = 1, self.size do
    for j = 1, #srcs do
        local sourceInput = srcs[j][b]:sub(2, srcs[j][b]:size(1)) -- skip first (ok for linescore since padded)
        self.sourceInput:sub(currRow, currRow+srcLen-1, 1, 1):copy(sourceInput)

        -- -- Source input is left padded [PPPPPPABCDE] .
        -- self.sourceInput[j][{{sourceOffset, self.sourceLength}, b}]:copy(sourceInput)
        -- self.sourceInputPadLeft = true

        if j <= 2*g_nRegRows then
            -- second feature is row name
            self.sourceInput:sub(currRow, currRow+srcLen-1, 2, 2):fill(srcs[j][b][1])
            -- third feature is col name
            self.sourceInput:sub(currRow, currRow+srcLen-1, 3, 3)
              :range(colStartIdx, colStartIdx+srcLen-1)
            -- fourth feature is home or away
            local lastFeat = j <= g_nRegRows and colStartIdx+2*srcLen or colStartIdx+2*srcLen+1
            self.sourceInput:sub(currRow, currRow+srcLen-1, 4, 4):fill(lastFeat)
        else
            self.sourceInput:sub(currRow, currRow+srcLen-1, 2, 2):fill(srcs[j][b][g_specPadding+1])
            self.sourceInput:sub(currRow, currRow+srcLen-1, 3, 3)
              :range(colStartIdx+srcLen, colStartIdx+2*srcLen-1)
            local lastFeat = j < #srcs and colStartIdx+2*srcLen or colStartIdx+2*srcLen+1
            self.sourceInput:sub(currRow, currRow+srcLen-1, 4, 4):fill(lastFeat)
        end
        currRow = currRow + srcLen
    end

    if tgt ~= nil then
      -- Input: [<s>ABCDE]
      -- Output: [ABCDE</s>]
      local targetLength = tgt[b]:size(1) - 1
      local targetInput = tgt[b]:narrow(1, 1, targetLength)
      local targetOutput = tgt[b]:narrow(1, 2, targetLength)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      self.targetInput[{{1, targetLength}, b}]:copy(targetInput)

      if targetMasks then
          for t = 1, targetLength do
              self.targetOutput[t][b][1] = targetOutput[t]
              if srcLocs[b][targetOutput[t]] then
                  local numInSrc = srcLocs[b][targetOutput[t]]:size(1)
                  self.targetOutput[t][b]:sub(2, numInSrc+1):copy(srcLocs[b][targetOutput[t]])
                  self.targetOutput[t][b][maxIndices+1] = numInSrc+1
              else
                  self.targetOutput[t][b][maxIndices+1] = 1
              end
          end
      else
          self.targetOutput[{{1, targetLength}, b}]:copy(targetOutput)
      end
    end
  end
  --print(currRow, self.sourceInput:size(1))
  assert(currRow == self.sourceInput:size(1)+1)
end


-- maps words to their (linearized) location (for each batch)
function findSourceLocations(srcs, batchSize, offset)
    local srcLocs = tds.Vec()
    for b = 1, batchSize do
        b_tbl = {}
        local srcIdx = 1
        for j = 1, #srcs do
            local sourceInput = srcs[j][b]:sub(2, srcs[j][b]:size(1))
            for t = 1, sourceInput:size(1) do
                if not b_tbl[sourceInput[t]] then
                    b_tbl[sourceInput[t]] = {}
                end
                table.insert(b_tbl[sourceInput[t]], srcIdx)
                srcIdx = srcIdx + 1
            end
        end
        -- copy into a tds thing
        local b_hash = tds.Hash()
        for k, v in pairs(b_tbl) do
            b_hash[k] = torch.IntTensor(v):add(offset)
        end
        srcLocs:insert(b_hash)
    end
    return srcLocs
end

-- -- would be faster to precompute everything for each minibatch, but might be tricky....
-- function getBatchLocations(srcLocs, tgt)
--     for b = 1, #tgt do
--         local targetLength = tgt[b]:size(1) - 1
--         local targetOutput = tgt[b]:narrow(1, 2, targetLength)
--         for t = 1, targetLength do
--             if srcLocs[b][targetOutput[t]] then


--[[ Set source input directly,

Parameters:

  * `sourceInput` - a Tensor of size (sequence_length, batch_size, feature_dim)
  ,or a sequence of size (sequence_length, batch_size). Be aware that sourceInput is not cloned here.

--]]
function BoxBatch2:setSourceInput(sourceInput)
    assert(false)
  assert (sourceInput:dim() >= 2, 'The sourceInput tensor should be of size (seq_len, batch_size, ...)')
  self.size = sourceInput:size(2)
  self.sourceLength = sourceInput:size(1)
  self.sourceInputFeatures = {}
  self.sourceInputRevReatures = {}
  self.sourceInput = sourceInput
  self.sourceInputRev = self.sourceInput:index(1, torch.linspace(self.sourceLength, 1, self.sourceLength):long())
  return self
end

--[[ Set target input directly.

Parameters:

  * `targetInput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD. Be aware that targetInput is not cloned here.
--]]
function BoxBatch2:setTargetInput(targetInput)
    assert(false)
  assert (targetInput:dim() == 2, 'The targetInput tensor should be of size (seq_len, batch_size)')
  self.targetInput = targetInput
  self.size = targetInput:size(2)
  self.totalSize = self.size
  self.targetLength = targetInput:size(1)
  self.targetInputFeatures = {}
  self.targetSize = torch.sum(targetInput:transpose(1,2):ne(onmt.Constants.PAD), 2):view(-1):double()
  return self
end

--[[ Set target output directly.

Parameters:

  * `targetOutput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD.  Be aware that targetOutput is not cloned here.
--]]
function BoxBatch2:setTargetOutput(targetOutput)
    assert(false)
  assert (targetOutput:dim() == 2, 'The targetOutput tensor should be of size (seq_len, batch_size)')
  self.targetOutput = targetOutput
  self.targetOutputFeatures = {}
  return self
end

local function addInputFeatures(inputs, featuresSeq, t)
  local features = {}
  for j = 1, #featuresSeq do
    table.insert(features, featuresSeq[j][t])
  end
  if #features > 1 then
    table.insert(inputs, features)
  else
    onmt.utils.Table.append(inputs, features)
  end
end

--[[ Get source batch at timestep `t`. --]]
function BoxBatch2:getSourceInput(t)
    assert(false)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.sourceInput[self.inputRow][t]

  if self.batchRowFeats then
      inputs = {inputs, self.batchRowFeats[self.inputRow], self.batchColFeats[t]}
  end

  -- if #self.sourceInputFeatures > 0 then
  --   inputs = { inputs }
  --   addInputFeatures(inputs, self.sourceInputFeatures, t)
  -- end

  return inputs
end

-- returns a nRows*srcLen x batchSize tensor
function BoxBatch2:getSource()
    return self.sourceInput
end

--[[ Get target input batch at timestep `t`. --]]
function BoxBatch2:getTargetInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.targetInput[t]

  return inputs
end

--[[ Get target output batch at timestep `t` (values t+1). --]]
function BoxBatch2:getTargetOutput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.targetOutput[t] }

  return outputs
end

return BoxBatch2
