# Imports all the necesary spacenames
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from numpy import argmax
from keras.callbacks import EarlyStopping

# Class in charge of handling the model
class MyModelHandler(object):
  # Receives all the necessary information to create the model
  def __init__(self, model_type, word_counts, tag_counts, embedding_size, units_size, input_len, show_summary=False, use_RNN=False, dense_len=0, char_counts=0, char_units_size=0, char_input_len=0, char_embedding_size=0):
    # Type of the model to be created (Secuencial/Funtional/Funtional_Bi)
    if model_type=="Secuencial":
      # Create a sequential model
      self.model = Sequential()
      # Input is the size of the vocabulary, output is the length of the vector of each represented word, Mark zero to skip the zeros (padded values)
      self.model.add(Embedding(word_counts, embedding_size, mask_zero=True))
      # The output of the LSTM must be a vector for each word, i.e. the hyperparameter return_sequences=True for the LSTM instance
      self.model.add(LSTM(units_size,return_sequences=True))
      # Time distributed layer with a dense layer inside with a softmax function for multiclassification
      self.model.add(TimeDistributed(Dense(tag_counts, activation=('softmax'))))

    # Implement the model using the functional API.
    elif model_type=="Functional":
      # Create the input of the functional api model with the length of the dictionary
      inputs = keras.Input(shape=(input_len,))
      # Input is the size of the vocabulary, output is the length of the vector of each represented word, Mark zero to skip the zeros (padded values)
      embedding_layer = layers.Embedding(word_counts,embedding_size,mask_zero=True)
      # Assing the inputs to the embedding layer to create the input layer 
      input_layer = embedding_layer(inputs)

      # If more than zero, add an extra dense layer to the model with the amount of neurons
      if (dense_len > 0):
        aux_layer = layers.Dense(dense_len)(input_layer)
      else:
        aux_layer = input_layer

      # A rnn (SimpleRNN or LSTM) based layer with return_sequences=True return the entire sequence of outputs for each sample (one vector per timestep per sample)
      # If a RNN or a LSTM layer should be used
      if (use_RNN):
        lstm_layer = layers.SimpleRNN(units_size,return_sequences=True)(aux_layer)
      else:
        # The output of the LSTM must be a vector for each word, i.e. the hyperparameter return_sequences=True for the LSTM instance
        lstm_layer = layers.LSTM(units_size,return_sequences=True)(aux_layer)
      
      # Time distributed layer with a dense layer inside with a softmax function for multiclassification
      outputs = layers.TimeDistributed(layers.Dense(tag_counts, activation=('softmax')))(lstm_layer)

      # Create the model
      self.model = keras.Model(inputs=inputs, outputs=outputs) 
    
    # Implement the model using the functional API with a bidirectional layer.
    elif model_type=="Functional_Bi":
      # Create the input of the functional api model with the length of the dictionary
      inputs = keras.Input(shape=(input_len,))

      # Input is the size of the vocabulary, output is the length of the vector of each represented word, Mark zero to skip the zeros (padded values)
      embedding_layer = layers.Embedding(word_counts,embedding_size,mask_zero=True)
      # Assing the inputs to the embedding layer to create the input layer 
      input_layer = embedding_layer(inputs)

      # If more than zero, add an extra dense layer to the model with the amount of neurons
      if (dense_len > 0):
        aux_layer = layers.Dense(dense_len)(input_layer)
      else:
        aux_layer = input_layer

      # A rnn (SimpleRNN or LSTM) based layer with return_sequences=True return the entire sequence of outputs for each sample (one vector per timestep per sample)
      # If a RNN or a LSTM layer should be used
      if (use_RNN):
        # Create a bidirectional layer with the RNN inside
        lstm_layer = layers.Bidirectional(layers.SimpleRNN(units_size,return_sequences=True))(aux_layer)
      else:
        # Create a bidirectional layer with the LSTM inside
        # The output of the LSTM must be a vector for each word, i.e. the hyperparameter return_sequences=True for the LSTM instance
        lstm_layer = layers.Bidirectional(layers.LSTM(units_size,return_sequences=True))(aux_layer)
     
      # Time distributed layer with a dense layer inside with a softmax function for multiclassification
      outputs = layers.TimeDistributed(layers.Dense(tag_counts, activation=('softmax')))(lstm_layer)
      
      # Create the model
      self.model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Implement the model using the functional API with a char aproximation.
    elif model_type=="Functional_Char":
      # Create the input of the functional api model with the length of the dictionary
      inputs = keras.Input(shape=(input_len,))
      # Input is the size of the vocabulary, output is the length of the vector of each represented word, Mark zero to skip the zeros (padded values)
      embedding_layer = layers.Embedding(word_counts,embedding_size,mask_zero=True)
      
      # A rnn (SimpleRNN or LSTM) based layer with return_sequences=True return the entire sequence of outputs for each sample (one vector per timestep per sample)
      # If a RNN or a LSTM layer should be used
      # LSTM layer by word
      if (use_RNN):
        # Create a bidirectional layer with the RNN inside
        lstm_layer = layers.Bidirectional(layers.SimpleRNN(units_size,return_sequences=True))(aux_layer)
      else:
        lstm_layer =layers.Bidirectional(layers.LSTM(units_size,return_sequences=True))

      # Time distributed layer with a dense layer inside with a softmax function for multiclassification
      dense = layers.TimeDistributed(layers.Dense(tag_counts, activation=('softmax')))
      
      # Create the input of the functional api model with the length of the dictionary and the length of the characters
      char_inputs = keras.Input(shape=(input_len,char_input_len,))
      # Embedding layer for the characters
      char_embedding_layer = layers.Embedding(char_counts,char_embedding_size,mask_zero=True)
      # LSTM layer for the caracters with time distributed
      if (use_RNN):
        # Create a bidirectional layer with the RNN inside
        char_level_LSTM = layers.TimeDistributed(layers.SimpleRNN(char_units_size,return_sequences=True))(aux_layer)
      else:
        # Dropout 0.2 to to help with overfitting in training
        char_level_LSTM = layers.TimeDistributed(layers.LSTM(char_units_size,return_sequences=False, dropout=0.2))

      # Create the final model
      y = char_embedding_layer(char_inputs)
      y = char_level_LSTM(y)
      x = embedding_layer(inputs)
      
      # Concatenate word and char model into one
      z = layers.Concatenate()([x, y])

      z = lstm_layer(z)
      outputs = dense(z)

      # Create the model
      self.model = keras.Model(inputs=[inputs,char_inputs], outputs=outputs)

    # If true show the summary of the model
    if show_summary:
      self.model.summary()
      
  # Trains and compile the model with the specified parameters
  def train(self, loss, optimizer, metrics, epochs, batch_size, train_ds, val_inputs, val_outputs, test_inputs, test_outputs):
    # Compile the model
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Monitor val_loss and if the minimun after three epochs keep incresing, stop the training
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1 , patience=3)

    # Train the model. Save the history for later use.
    self.history = self.model.fit(train_ds,validation_data=(val_inputs, val_outputs), epochs=epochs, batch_size=batch_size, callbacks=[es])

    #Evaluate accuracy against the test data
    loss, accuracy = self.model.evaluate(test_inputs, test_outputs)
    
  # Trains and compile the char model with the specified parameters
  def train_char(self, loss, optimizer, metrics, epochs, batch_size, train_inputs_words, train_inputs_chars, train_outputs, val_inputs_words, val_inputs_chars, val_outputs, test_inputs_words, test_inputs_chars, test_outputs):
    # Compile the model
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Monitor val_loss and if the minimun after three epochs keep incresing, stop the training
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1 , patience=3)

    # Train the model. Save the history for later use.
    self.history = self.model.fit(x=[train_inputs_words, train_inputs_chars], y=train_outputs,validation_data=([val_inputs_words,val_inputs_chars], val_outputs), epochs=epochs, batch_size=batch_size, callbacks=[es])
    
    #Evaluate accuracy against the test data
    loss, accuracy = self.model.evaluate([test_inputs_words,test_inputs_chars],test_outputs)
    
  # Evaluates the real accuracy of the upos comparing the desired with the prediction
  def evaluate_predict(self, test_inputs, test_outputs, max_len):
    # predict the outputs for the inputs
    predicted_outputs =  self.model.predict(test_inputs)
    
    tot = 0
    ok = 0
    # check every upos prediction with every desired upos
    for x in range(0, len(predicted_outputs)):
      for i in range(max_len):
        # Use argmax to pass to binary the float result
        t = argmax(test_outputs[x][i])
        
        # If is not a padding
        if (t > 0):
          tot=tot+1
          
          if (t == argmax(predicted_outputs[x][i])):
            ok=ok+1

    return (tot, ok)

  # Evaluates the real accuracy of the upos comparing the desired with the prediction of the advanced model
  def evaluate_predict_char(self, test_inputs_words, test_inputs_chars, test_outputs, max_len):
    # predict the outputs for the inputs
    predicted_outputs =  self.model.predict([test_inputs_words, test_inputs_chars])
    
    tot = 0
    ok = 0
    # check every upos prediction with every desired upos
    for x in range(0, len(predicted_outputs)):
      for i in range(max_len):
        # Use argmax to pass to binary the float result
        t = argmax(test_outputs[x][i])
        
        # If is not a padding
        if (t > 0):
          tot=tot+1
          
          if (t == argmax(predicted_outputs[x][i])):
            ok=ok+1

    return (tot, ok)

  # Test the prediction of a text
  def test_predict(self, text_to_predict_tuple):
    # predict the outputs for the inputs
    predicted_outputs =  self.model.predict(text_to_predict_tuple[0])

    res = []
    for x in predicted_outputs[0]:
      # Use argmax to pass to binary the float result
      res.append(argmax(x))

    # Return only the real values (without padding)
    return res[:text_to_predict_tuple[1]]
    
  # Test the prediction of a text of the advanced model
  def test_predict_char(self, text_to_predict_tuple):
    # predict the outputs for the inputs
    predicted_outputs =  self.model.predict([text_to_predict_tuple[1],text_to_predict_tuple[2]])

    res = []
    for x in predicted_outputs[0]:
      # Use argmax to pass to binary the float result
      res.append(argmax(x))

    res_tot = []
    for i in range(len(predicted_outputs)):
      res_sen = []
      for ii in range(len(text_to_predict_tuple[0][i])):
        # Use argmax to pass to binary the float result
        res = argmax(predicted_outputs[i][ii])
        res_sen.append((text_to_predict_tuple[0][i][ii], res))
      
      res_tot.append(res_sen)

    # Return only the real values (without padding)
    return res_tot

  # Gets the fit history execution
  def get_history(self):
    return self.history

   