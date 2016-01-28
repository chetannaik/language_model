require 'torch'
require 'nn'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data','data directory(contain the file input.txt with input data and dictionary.t7)')
cmd:option('-start_token','$START$','token used to denote start of sentence.')
cmd:option('-end_token','$END$','token used to denote start of sentence.')
-- model params
cmd:option('-models_dir','models','models directory')
cmd:option('-vector_size',25,'size of the word vector')
cmd:option('-window_size',5,'size of the centext window')
-- optimization
cmd:option('-batch_size',5,'number of sequences to train on in parallel')
cmd:option('-epochs',2,'number of full passes through the training data')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
local dictionary_file = path.join(opt.data_dir, 'dictionary.t7')
torch.manualSeed(opt.seed)

local word_dict = {}

-- length of a table
function table_length(T)
  local count = 0
  for _ in pairs(T) do
    count = count + 1
  end
  return count
end

-- initialize using word2vec vectors
function initialize_embeddings(data_dir, load_dict)
  local w2vutils = require 'utils.w2vutils'
  local vocabulary_file_path = path.join(data_dir, 'vocabulary.txt')
  local vocabulary_file, err = io.open(vocabulary_file_path, "r")
  if err then
    print(err)
    return err
  end

  if load_dict then
    local d_file = io.open(dictionary_file, "r")
    if err then print(err) end
    if d_file ~= nil then
      word_dict = torch.load(dictionary_file)
      d_file:close()
    end
  end

  while true do
    local word = vocabulary_file:read()
    if not word then break end
    if not word_dict[word] then
      local google_vec = w2vutils:word2vec(word)
      if not google_vec then
        google_vec = torch.randn(opt.vector_size)
      end
      google_vec = google_vec:narrow(1, 1, opt.vector_size)
      word_dict[word] = google_vec
    end
  end
  w2vutils = nil
  torch.save(dictionary_file, word_dict)
end


-- read a small batch
function read_batch(train_file)
  local train_data_count = 1
  local batch_train = {}
  local words = {}
  word_dict = torch.load(dictionary_file)

  while train_data_count <= opt.batch_size do
    local train_data = {}
    local p_example = train_file:read()
    local n_example = train_file:read()

    if not p_example or not n_example then break end

    local start_idx = 1
    local end_idx = opt.vector_size

    p_split = string.split(p_example, " ")
    n_split = string.split(n_example, " ")
    local mid_word = math.ceil(#p_split / 2)

    for w_idx = 1, #p_split do
      local word = p_split[w_idx]
      local pos_word_vec = word_dict[word]
      for idx = start_idx, end_idx do
          train_data[idx] = pos_word_vec[idx - start_idx + 1]
      end
      if w_idx == mid_word then table.insert(words, word) end
      start_idx = end_idx + 1
      end_idx = start_idx + opt.vector_size - 1
    end

    batch_train[2 * train_data_count - 1] = {torch.Tensor(train_data), 1}

    start_idx = 1
    end_idx = opt.vector_size

    for w_idx =1, #n_split do
      local word = n_split[w_idx]
      local neg_word_vec = word_dict[word]
      for idx = start_idx, end_idx do
        train_data[idx] = neg_word_vec[idx - start_idx + 1]
      end
      if w_idx == mid_word then table.insert(words, word) end
      start_idx = end_idx + 1
      end_idx = start_idx + opt.vector_size - 1
    end

    batch_train[2 * train_data_count] = {torch.Tensor(train_data), 2}

    train_data_count = train_data_count + 1
  end

  train_data_count = train_data_count - 1

  function batch_train:size() return train_data_count * 2 end

  return  batch_train, words
end


-- train
function train_lm(epoch)
  local train_data_path = path.join(opt.data_dir, 'train_data.txt')
  -- construct nn
  net = nn.Sequential()
  -- add layers
  local inputs = opt.window_size * opt.vector_size;
  local outputs = 2;
  local HUs = 50;

  net:add(nn.Linear(inputs, HUs))
  net:add(nn.Sigmoid())
  net:add(nn.Linear(HUs, outputs))
  net:add(nn.LogSoftMax())

  -- define loss function
  local criterion = nn.ClassNLLCriterion()

  local trainer = nn.StochasticGradient(net, criterion)
  trainer.learningRate = 0.01
  trainer.maxIteration = 1

  for e = 1, epoch do
    print('Starting iteration:', e)
    local train_data = io.open(train_data_path)
    -- Run Batch Training and Gradient descend
    while true do
      local batch_train_data, words = read_batch(train_data)
      if batch_train_data:size() == 0 then break end

      -- closure for editing word vector.
      function edit_vector(batch_data_idx)
        local word = words[batch_data_idx]
        local word_vec = word_dict[word]
        local gradIpOffset = ((math.floor(opt.window_size / 2) + 1) * opt.vector_size)
        for idx = 1, opt.vector_size do
          word_vec[idx] = net.gradInput[gradIpOffset + idx]
          --word_vec[idx] = word_vec[idx] - word_vec[idx] * net.gradInput[gradIpOffset + idx]
        end
        word_dict[word] = word_vec
      end

      trainer:train(batch_train_data, edit_vector)
      torch.save(dictionary_file, word_dict)
    end
  end
end

-- remove old files
function clean_files()
  os.remove(dictionary_file)
end

function main()
  clean_files()
  initialize_embeddings(opt.data_dir, true)
  train_lm(opt.epochs)
end

main()
