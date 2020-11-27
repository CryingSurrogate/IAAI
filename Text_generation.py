#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/CryingSurrogate/IAAI/blob/main/Text_generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Text generation with an RNN

# ## Setup

# ### Import TensorFlow and other libraries

# In[2]:


import tensorflow as tf

import numpy as np
import os
import time


# ### Download the Shakespeare dataset
# 
# Change the following line to run this code on your own data.

# In[3]:


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://www.dropbox.com/s/ize6bmgn78e5jju/IA.txt?dl=1')


# ### Read the data
# 
# First, look in the text:

# In[4]:


# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))


# In[5]:


# Take a look at the first 250 characters in text
print(text[:250])


# In[6]:


# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))


# ## Process the text

# ### Vectorize the text
# 
# Before training, you need to map strings to a numerical representation. Create two lookup tables: one mapping characters to numbers, and another for numbers to characters.

# In[7]:


# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


# Now you have an integer representation for each character. Notice that you mapped the character as indexes from 0 to `len(unique)`.

# In[8]:


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# In[9]:


# Show how the first 13 characters from the text are mapped to integers
print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


# ### The prediction task

# Given a character, or a sequence of characters, what is the most probable next character? This is the task you're training the model to perform. The input to the model will be a sequence of characters, and you train the model to predict the output—the following character at each time step.
# 
# Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment, what is the next character?
# 

# ### Create training examples and targets
# 
# Next divide the text into example sequences. Each input sequence will contain `seq_length` characters from the text.
# 
# For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.
# 
# So break the text into chunks of `seq_length+1`. For example, say `seq_length` is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".
# 
# To do this first use the `tf.data.Dataset.from_tensor_slices` function to convert the text vector into a stream of character indices.

# In[10]:


# The maximum length sentence you want for a single input in characters
seq_length = 60
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])


# The `batch` method lets us easily convert these individual characters to sequences of the desired size.

# In[11]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


# For each sequence, duplicate and shift it to form the input and target text by using the `map` method to apply a simple function to each batch:

# In[12]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# Print the first example input and target values:

# In[13]:


for input_example, target_example in  dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# Each index of these vectors is processed as a one time step. For the input at time step 0, the model receives the index for "F" and tries to predict the index for "i" as the next character. At the next timestep, it does the same thing but the `RNN` considers the previous step context in addition to the current input character.

# In[14]:


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# ### Create training batches
# 
# You used `tf.data` to split the text into manageable sequences. But before feeding this data into the model, you need to shuffle the data and pack it into batches.

# In[15]:


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# ## Build The Model

# Use `tf.keras.Sequential` to define the model. For this simple example three layers are used to define our model:
# 
# * `tf.keras.layers.Embedding`: The input layer. A trainable lookup table that will map the numbers of each character to a vector with `embedding_dim` dimensions;
# * `tf.keras.layers.GRU`: A type of RNN with size `units=rnn_units` (You can also use an LSTM layer here.)
# * `tf.keras.layers.Dense`: The output layer, with `vocab_size` outputs.

# In[16]:


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


# In[17]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


# In[18]:


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


# For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input, and applies the dense layer to generate logits predicting the log-likelihood of the next character:
# 
# ![A drawing of the data passing through the model](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/text_generation_training.png?raw=1)

# Please note that Keras sequential model is used here since all the layers in the model only have single input and produce single output. In case you want to retrieve and reuse the states from stateful RNN layer, you might want to build your model with Keras functional API or model subclassing. Please check [Keras RNN guide](https://www.tensorflow.org/guide/keras/rnn#rnn_state_reuse) for more details.

# ## Try the model
# 
# Now run the model to see that it behaves as expected.
# 
# First check the shape of the output:

# In[19]:


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# In the above example the sequence length of the input is `100` but the model can be run on inputs of any length:

# In[20]:


model.summary()


# To get actual predictions from the model you need to sample from the output distribution, to get actual character indices. This distribution is defined by the logits over the character vocabulary.
# 
# Note: It is important to _sample_ from this distribution as taking the _argmax_ of the distribution can easily get the model stuck in a loop.
# 
# Try it for the first example in the batch:

# In[21]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


# This gives us, at each timestep, a prediction of the next character index:

# In[22]:


sampled_indices


# Decode these to see the text predicted by this untrained model:

# In[23]:


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# ## Train the model

# At this point the problem can be treated as a standard classification problem. Given the previous RNN state, and the input this time step, predict the class of the next character.

# ### Attach an optimizer, and a loss function

# The standard `tf.keras.losses.sparse_categorical_crossentropy` loss function works in this case because it is applied across the last dimension of the predictions.
# 
# Because your model returns logits, you need to set the `from_logits` flag.
# 

# In[24]:


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# Configure the training procedure using the `tf.keras.Model.compile` method. Use `tf.keras.optimizers.Adam` with default arguments and the loss function.

# In[25]:


model.compile(optimizer='adam', loss=loss)


# ### Configure checkpoints

# Use a `tf.keras.callbacks.ModelCheckpoint` to ensure that checkpoints are saved during training:

# In[26]:


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# ### Execute the training

# To keep training time reasonable, use 10 epochs to train the model. In Colab, set the runtime to GPU for faster training.

# In[27]:


EPOCHS = 82


# In[28]:


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# ## Generate text

# ### Restore the latest checkpoint

# To keep this prediction step simple, use a batch size of 1.
# 
# Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.
# 
# To run the model with a different `batch_size`, you need to rebuild the model and restore the weights from the checkpoint.
# 

# In[ ]:


tf.train.latest_checkpoint(checkpoint_dir)


# In[ ]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# In[ ]:


model.summary()


# ### The prediction loop
# 
# The following code block generates the text:
# 
# * Begin by choosing a start string, initializing the RNN state and setting the number of characters to generate.
# 
# * Get the prediction distribution of the next character using the start string and the RNN state.
# 
# * Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.
# 
# * The RNN state returned by the model is fed back into the model so that it now has more context, instead of only one character. After predicting the next character, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted characters.
# 
# 
# ![To generate text the model's output is fed back to the input](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/images/text_generation_sampling.png?raw=1)
# 
# Looking at the generated text, you'll see the model knows when to capitalize, make paragraphs and imitates a Shakespeare-like writing vocabulary. With the small number of training epochs, it has not yet learned to form coherent sentences.

# In[ ]:


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# In[ ]:


print(generate_text(model, start_string=u"King of"))


# The easiest thing you can do to improve the results is to train it for longer (try `EPOCHS = 30`).
# 
# You can also experiment with a different start string, try adding another RNN layer to improve the model's accuracy, or adjust the temperature parameter to generate more or less random predictions.

# ## Advanced: Customized Training
# 
# The above training procedure is simple, but does not give you much control.
# 
# So now that you've seen how to run the model manually let's unpack the training loop, and implement it ourselves. This gives a starting point if, for example, you want to implement _curriculum learning_ to help stabilize the model's open-loop output.
# 
# Use `tf.GradientTape` to track the gradients. You can learn more about this approach by reading the [eager execution guide](https://www.tensorflow.org/guide/eager).
# 
# The procedure works as follows:
# 
# * First, reset the RNN state. You do this by calling the `tf.keras.Model.reset_states` method.
# 
# * Next, iterate over the dataset (batch by batch) and calculate the *predictions* associated with each.
# 
# * Open a `tf.GradientTape`, and calculate the predictions and loss in that context.
# 
# * Calculate the gradients of the loss with respect to the model variables using the `tf.GradientTape.grads` method.
# 
# * Finally, take a step downwards by using the optimizer's `tf.train.Optimizer.apply_gradients` method.
# 

# In[ ]:


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


# In[ ]:


optimizer = tf.keras.optimizers.Adam()


# In[ ]:


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# In[ ]:


# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # resetting the hidden state at the start of every epoch
    model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch + 1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

