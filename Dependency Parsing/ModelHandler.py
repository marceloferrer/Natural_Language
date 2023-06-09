# Imports all the necesary spacenames
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Embedding
from keras.callbacks import EarlyStopping

# Class in charge of handling the model creation
class MyModelHandler:
  # Receives all the necessary information to create the models
  def __init__(self, max_len, max_len_pre, char_len_pre, word_count, upos_count, action_count, relation_count, tag_count_pre):
    self.max_len = max_len
    self.max_len_pre = max_len_pre
    self.char_len_pre = char_len_pre
    self.word_count = word_count
    self.upos_count = upos_count
    self.action_count = action_count
    self.relation_count = relation_count
    self.tag_count_pre = tag_count_pre

  # Receives the type of model
  def get_model(self, model_type):
    if model_type=="Basic":
      # Input Layer stack
      inputs_stack = keras.Input(shape=(self.max_len,))
      # Input layer buffer
      inputs_buffer = keras.Input(shape=(self.max_len,))

      # Embedding layer for stack. Mask zero to ignore zeroes. Turns positive integers (indexes) into dense vectors of fixed size.
      embedding_layer_stack = Embedding(self.word_count,self.max_len,mask_zero=True)(inputs_stack)
      # Embedding layer for buffer. Mask zero to ignore zeroes. Turns positive integers (indexes) into dense vectors of fixed size.
      embedding_layer_buffer = Embedding(self.word_count,self.max_len,mask_zero=True)(inputs_buffer)

      # Concatenate both embeddings
      conct = layers.Concatenate()([embedding_layer_stack, embedding_layer_buffer])

      # Kernel slides along one dimension.
      x = layers.Conv1D(filters=128, kernel_size=3)(conct)
      # Downsamples the input representation.
      x = layers.GlobalMaxPooling1D()(x)

      # Output of the model. Softmax for multiclasiffication.
      outputs_1 = layers.Dense(self.action_count, activation=('softmax'))(x)
      outputs_2 = layers.Dense(self.relation_count, activation=('softmax'))(x)

      # Instance the model
      return keras.Model(inputs=[inputs_stack, inputs_buffer], outputs=[outputs_1, outputs_2])

    elif model_type=="Advanced":
      # Input Layer stack words
      inputs_stack = keras.Input(shape=(self.max_len,))
      # Input layer buffer words
      inputs_buffer = keras.Input(shape=(self.max_len,))
      # Input Layer stack upos
      inputs_stack_pos = keras.Input(shape=(self.max_len,))
      # Input layer buffer upos
      inputs_buffer_pos = keras.Input(shape=(self.max_len,))

      # Embedding layer for stack. Mask zero to ignore zeroes. Turns positive integers (indexes) into dense vectors of fixed size.
      embedding_layer_stack = Embedding(self.word_count,self.max_len,mask_zero=True)
      # Embedding layer for buffer. Mask zero to ignore zeroes. Turns positive integers (indexes) into dense vectors of fixed size.
      embedding_layer_buffer = Embedding(self.word_count,self.max_len,mask_zero=True)
      # Embedding layer for upos stack. Mask zero to ignore zeroes. Turns positive integers (indexes) into dense vectors of fixed size.
      embedding_layer_stack_pos = Embedding(self.upos_count,self.max_len,mask_zero=True)
      # Embedding layer for upos buffer. Mask zero to ignore zeroes. Turns positive integers (indexes) into dense vectors of fixed size.
      embedding_layer_buffer_pos = Embedding(self.upos_count,self.max_len,mask_zero=True)

      # Kernel slides along one dimension.
      conv1D = layers.Conv1D(128,3)
      # Downsamples the input representation.
      maxpool= layers.GlobalMaxPooling1D()

      # Output of the model. Softmax for multiclasiffication.
      dense1=Dense(self.action_count, activation=('softmax'))
      dense2=Dense(self.relation_count, activation=('softmax'))

      # Ensemble the model
      x=embedding_layer_stack(inputs_stack)
      y=embedding_layer_buffer(inputs_buffer)
      r=embedding_layer_stack_pos(inputs_stack_pos)
      s=embedding_layer_buffer_pos(inputs_buffer_pos)
      z=layers.Concatenate(axis=1)([x, y,r,s])
      z=conv1D(z)
      z=maxpool(z)
      output1=dense1(z)
      output2=dense2(z)

      # Instance the model
      return keras.Model(inputs=[inputs_stack,inputs_buffer,inputs_stack_pos,inputs_buffer_pos], outputs=[output1, output2])
    
    # Implement the model using the functional API with a bidirectional layer.
    elif model_type=="Advanced_Pre":
      # Create the input of the functional api model with the length of the dictionary
      inputs = keras.Input(shape=(self.max_len_pre,))
      # Input is the size of the vocabulary, output is the length of the vector of each represented word, Mark zero to skip the zeros (padded values)
      embedding_layer = layers.Embedding(self.word_count,self.max_len_pre,mask_zero=True)

      # Create a bidirectional layer with the LSTM inside
      # The output of the LSTM must be a vector for each word, i.e. the hyperparameter return_sequences=True for the LSTM instance
      lstm_layer =layers.Bidirectional(layers.LSTM(self.max_len_pre,return_sequences=True))

      # Time distributed layer with a dense layer inside with a softmax function for multiclassification
      dense = layers.TimeDistributed(layers.Dense(self.tag_count_pre, activation=('softmax')))

      # Create the input of the functional api model with the length of the dictionary and the length of the characters
      char_inputs = keras.Input(shape=(self.max_len_pre,self.char_len_pre))
      # Embedding layer for the characters
      char_embedding_layer = layers.Embedding(self.char_len_pre,self.max_len_pre,mask_zero=True)

      # Create a bidirectional layer with the LSTM inside. return_sequences=False for the use of chars
      char_level_LSTM = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(self.max_len_pre,return_sequences=False)))

      # Create the final model
      y = char_embedding_layer(char_inputs)
      y = char_level_LSTM(y)
      x = embedding_layer(inputs)

      # Concatenate word and char model into one
      z = layers.Concatenate()([x, y])

      z = lstm_layer(z)
      outputs = dense(z)

      # Create the model
      return keras.Model(inputs=[inputs,char_inputs], outputs=outputs)      
 