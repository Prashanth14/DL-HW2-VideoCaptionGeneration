import tensorflow as tf
import numpy as np
import random as rd

class Seq2Seq_Model():
    def __init__(self, nNetSize, nLayer, featureDim, embeddingSize, lambdaR, wordKeyTrans, mode, maxGradNorm, useAttention, beamSearch, beamSize, maxEncoderSteps, maxDecoderSteps):
        tf.set_random_seed(2000)
        np.random.seed(2000)
        rd.seed(2000)

        self.nNetSize = nNetSize
        self.nLayer = nLayer
        self.featureDim = featureDim
        self.embeddingSize = embeddingSize
        self.lambdaR = lambdaR
        self.wordKeyTrans = wordKeyTrans
        self.mode = mode
        self.maxGradNorm = maxGradNorm
        self.useAttention = useAttention
        self.beamSearch = beamSearch
        self.beamSize = beamSize
        self.maxEncoderSteps = maxEncoderSteps
        self.maxDecoderSteps = maxDecoderSteps
        self.vocab_size = len(self.wordKeyTrans)

        self.buildModel()

    def CreateRnnCell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.GRUCell(self.nNetSize)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder, seed=2020)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.nLayer)])
        return cell

    def buildModel(self):
        tf.set_random_seed(2000)
        np.random.seed(2000)
        rd.seed(2000)

        print ('Building model...')
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        ###################### Define model's encoder #############################
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # Encoder embedding.
            encoder_inputs_flatten = tf.reshape(self.encoder_inputs, [-1, self.featureDim])
            encoder_inputs_embedded = tf.layers.dense(encoder_inputs_flatten, self.embeddingSize, use_bias=True)
            encoder_inputs_embedded = tf.reshape(encoder_inputs_embedded, [self.batch_size, self.maxEncoderSteps, self.nNetSize])

            # Building RNN cell
            encoder_cell = self.CreateRnnCell()

            # Run Dynamic RNN
            #   encoder_outputs: [batch_size, max_time, num_units]
            #   encoder_state: [batch_size, num_units]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embedded, 
                sequence_length=self.encoder_inputs_length, 
                dtype=tf.float32)

        # ========== Define model's encoder ==========        
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            encoder_inputs_length = self.encoder_inputs_length

            if self.beamSearch:
                # If you use beamSearch, you need to tile_batch the output of the encoder, which is actually copying beamSize
                print("Using beamsearch decoding...")
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beamSize)
                encoder_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beamSize), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beamSize)

            #If beam_seach is used, batch_size = self.batch_size * self.beamSize. Because it has been copied once before
            batch_size = self.batch_size if not self.beamSearch else self.batch_size * self.beamSize

            # A dense matrix to turn the top hidden states to logit vectors of dimension V.
            projection_layer = tf.layers.Dense(units=self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=2020))
            
            # Decoder embedding
            embedding_decoder = tf.Variable(tf.random_uniform([self.vocab_size, self.nNetSize], -0.1, 0.1, seed=2020), name='embedding_decoder')


            ####################Build RNN cell##################################
            decoder_cell = self.CreateRnnCell()

            if self.useAttention:
                #Define the attention mechanism to be used
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.nNetSize, 
                    memory=encoder_outputs, 
                    normalize=True,
                    memory_sequence_length=encoder_inputs_length)

                #Define the LSTMCell used in the decoder stage, then encapsulate the attention wrapper
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, 
                    attention_mechanism=attention_mechanism, 
                    attention_layer_size=self.nNetSize, 
                    name='Attention_Wrapper')

                #Define the initialization state of the decoder stage, directly use the last hidden layer state of the encoder stage for assignment
                decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=2020))

           
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], self.wordKeyTrans['<bos>']), ending], 1)
            
            decoder_inputs_embedded = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

            # Helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_inputs_embedded, 
                sequence_length=self.decoder_targets_length, 
                time_major=False, name='training_helper')
            # Decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper, 
                initial_state=decoder_initial_state, 
                output_layer=output_layer)
            
            
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, impute_finished=True, maximum_iterations=self.max_target_sequence_length)

            # Calculate the loss and gradient according to the output, and define the AdamOptimizer and train_op to be updated
            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')

            # Use sequence_loss to calculate loss, here you need to pass in the mask flag defined previously
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.decoder_logits_train, 
                targets=self.decoder_targets, 
                weights=self.mask)

            # Training summary for the current batch_loss
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(self.lambdaR)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradNorm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

           
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.wordKeyTrans['<bos>']
            end_token = self.wordKeyTrans['<eos>']
            
            
            if self.beamSearch:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell, 
                    embedding=embedding_decoder,
                    start_tokens=start_tokens, 
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beamSize,
                    output_layer=output_layer)
            else:
                # Helper
                inference_decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding_decoder, 
                    start_tokens=start_tokens, 
                    end_token=end_token)
                # Decoder
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell, 
                    helper=inference_decoding_helper, 
                    initial_state=decoder_initial_state, 
                    output_layer=output_layer)

          
            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=self.maxDecoderSteps)

            if self.beamSearch:
                self.decoder_predict_decode = inference_decoder_outputs.predicted_ids
                self.decoder_predict_logits = inference_decoder_outputs.beam_search_decoder_output
            else:
                self.decoder_predict_decode = tf.expand_dims(inference_decoder_outputs.sample_id, -1)
                self.decoder_predict_logits = inference_decoder_outputs.rnn_output

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
        #For the training phase, you need to execute three ops of self.train_op, self.loss, self.summary_op, and pass in the corresponding data
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.decoder_targets: decoder_targets,
                      self.decoder_targets_length: decoder_targets_length,
                      self.keep_prob_placeholder: 0.5,
                      self.batch_size: len(encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
        #For the eval stage, there is no need to backpropagate, so only execute self.loss, self.summary_op two ops, and pass in the corresponding data
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.decoder_targets: decoder_targets,
                      self.decoder_targets_length: decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, encoder_inputs, encoder_inputs_length):
        #The infer stage only needs to run the final result, and does not need to calculate the loss, so feed_dict only needs to pass in the corresponding data of encoder_input
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(encoder_inputs)}
        predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict=feed_dict)
        return predict, logits
