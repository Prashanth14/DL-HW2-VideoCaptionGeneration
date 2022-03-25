#Run train file using following command
# python train.py /Users/manish/Downloads/MLDS_hw2_data/testing_data/feat/ ./Users/manish/Downloads/MLDS_hw2_data/testing_label.json ./output_testset_manish.txt 

import os
import sys
import time
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import random as rd
import pickle as pk
import warnings

from bleuEval import BLEU
from sequence import pad_seqs as ps
from seq2SeqModel import Seq2Seq_Model
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    np.random.seed(2000)
    rd.seed(2000)
    
    
    test_feat_folder = sys.argv[1]
    testing_label_json = sys.argv[2]
    output_testset = sys.argv[3]

    tf.app.flags.DEFINE_integer('nnet_size', 1024, 'Number of hidden units per layer')
    tf.app.flags.DEFINE_integer('n_layer', 2, 'Number of layers per encoder and decoder')
    tf.app.flags.DEFINE_integer('feature_dim', 4096, 'Feature dimensions per video frame')
    tf.app.flags.DEFINE_float('lambda_r', 0.001, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
    tf.app.flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
    
    tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'Maximum gradient norm')
    
    tf.app.flags.DEFINE_integer('sample_size', 1450, 'train data sample')
    tf.app.flags.DEFINE_integer('frame_dim', 80, '# of frame per video')

    tf.app.flags.DEFINE_boolean('use_attention', True, 'Attention Enabled')  
    
    tf.app.flags.DEFINE_boolean('beam_search', False, 'Beam search Disabled')
    tf.app.flags.DEFINE_integer('beam_size', 5, 'Size of beam search')
    
    tf.app.flags.DEFINE_integer('max_encoder_steps', 64, 'Maximum encoder steps')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Maximum decoder steps')
    
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'model directory')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'Checkpoints file name')

    FLAGS = tf.app.flags.FLAGS
##################################################################################################################
    num_top_BLEU = 10 #number considered
    
    top_BLEU = list()

    print ('Printing pickle generated file:')
    
    wordkeytrans = pk.load(open('wordkeytrans.obj', 'rb'))
    keywordtrans = pk.load(open('keywordtrans.obj', 'rb'))
    video_IDs = pk.load(open('vid_id.obj', 'rb'))
    dict_caption = pk.load(open('dict_caption.obj', 'rb'))
    dict_feat = pk.load(open('dict_feat.obj', 'rb'))
    keywordtrans_srs = pd.Series(keywordtrans)

    
    test_feat_file = os.listdir(test_feat_folder) #we will be using listdir for pathof feat in test file
    test_feat_path = [(test_feat_folder + filename) for filename in test_feat_file] #we will be generating filename extension
    test_video_IDs = [filename[:-4] for filename in test_feat_file] # we will be removing filename extension .avi

    test_dict_feat = {} #A dictionary for storing video id, feature of test set
    
    for path in test_feat_path:
        test_video_feat = np.load(path) #loading path
        
        sampled_video_frame = sorted(rd.sample(range(FLAGS.frame_dim), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = path[: -4].replace(test_feat_folder, "")
        test_dict_feat[test_video_ID] = test_video_feat
    

    test_vid_caption = json.load(open(testing_label_json, 'r'))

    with tf.Session() as sess:
        model = Seq2Seq_Model(nnet_size=FLAGS.nnet_size, n_layer=FLAGS.n_layer, feature_dim=FLAGS.feature_dim, embedding_size=FLAGS.embedding_size, 
            lambda_r=FLAGS.lambda_r, wordkeytrans=wordkeytrans, mode='train', max_grad_norm=FLAGS.max_grad_norm, use_attention=FLAGS.use_attention, 
            beam_search=FLAGS.beam_search, beam_size=FLAGS.beam_size, max_encoder_steps=FLAGS.max_encoder_steps, max_decoder_steps=FLAGS.max_decoder_steps)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Model Reload is done ')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Building a new model')
            sess.run(tf.global_variables_initializer())

        summary_file = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()
            
            sampled_ID_caption = list()# Random sample ID_caption.
            for ID in video_IDs:
                sampled_caption = rd.sample(dict_caption[ID], 1)[0]
                sampled_video_frame = sorted(rd.sample(range(FLAGS.frame_dim), FLAGS.max_encoder_steps))
                sampled_video_feat = dict_feat[ID][sampled_video_frame]
                sampled_ID_caption.append((sampled_video_feat, sampled_caption))

            
            rd.shuffle(sampled_ID_caption)#Shuffling the training set

            for batch_start, batch_end in zip(range(0, FLAGS.sample_size, FLAGS.batch_size), range(FLAGS.batch_size, FLAGS.sample_size, FLAGS.batch_size)):
                print ("Training done for batch:%04d/%04d" %(batch_end, FLAGS.sample_size))

                batch_sampled_ID_caption = sampled_ID_caption[batch_start : batch_end]
                batch_video_feats = [elements[0] for elements in batch_sampled_ID_caption]
                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size
                # batch_video_feat_mask = np.zeros((batch_size, max_encoder_steps))
                batch_captions = np.array(["<bos> "+ elements[1] for elements in batch_sampled_ID_caption])

                for index, caption in enumerate(batch_captions):
                    caption_words = caption.lower().split(" ")
                    if len(caption_words) < FLAGS.max_decoder_steps:
                        batch_captions[index] = batch_captions[index] + " <eos>"
                    else:
                        new_caption = ""
                        for i in range(FLAGS.max_decoder_steps - 1):
                            new_caption = new_caption + caption_words[i] + " "
                        batch_captions[index] = new_caption + "<eos>"

                batch_captions_words_index = list()
                for caption in batch_captions:
                    words_index = list()
                    for caption_words in caption.lower().split(' '):
                        if caption_words in wordkeytrans:
                            words_index.append(wordkeytrans[caption_words])
                        else:
                            words_index.append(wordkeytrans['<unk>'])
                    batch_captions_words_index.append(words_index)

                batch_captions_matrix = ps(batch_captions_words_index, pad_str='post', max_len=FLAGS.max_decoder_steps)
                batch_captions_length = [len(x) for x in batch_captions_matrix]
               
                loss, summary = model.train(sess, batch_video_feats, batch_video_frame, batch_captions_matrix, batch_captions_length)            
               
            ##########Validation on test data set##########

            test_caption_list = list()
            for batch_start, batch_end in zip(range(0, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, FLAGS.sample_size))
                if batch_end < len(test_video_IDs):
                    batch_sampled_ID = np.array(test_video_IDs[batch_start : batch_end])
                    batch_video_feats = [test_dict_feat[x] for x in batch_sampled_ID]
                else:
                    batch_sampled_ID = test_video_IDs[batch_start : batch_end]
                    for _ in range(batch_end - len(test_video_IDs)):
                        batch_sampled_ID.append(test_video_IDs[-1])
                    batch_sampled_ID = np.array(batch_sampled_ID)
                    batch_video_feats = [test_dict_feat[x] for x in batch_sampled_ID]

                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size 

                batch_caption_words_index, logits = model.infer(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame) 

                if batch_end < len(test_video_IDs):
                    batch_caption_words_index = batch_caption_words_index
                else:
                    batch_caption_words_index = batch_caption_words_index[:len(test_video_IDs) - batch_start]

                for index, test_caption_words_index in enumerate(batch_caption_words_index):
                    

                    if FLAGS.beam_search:
                        logits = np.array(logits).reshape(-1, FLAGS.beam_size)
                        max_logits_index = np.argmax(np.sum(logits, axis=0))
                        predict_list = np.ndarray.tolist(test_caption_words_index[0, :, max_logits_index])
                        predict_seq = [keywordtrans[idx] for idx in predict_list]
                        test_caption_words = predict_seq
                    else:
                        test_caption_words_index = np.array(test_caption_words_index).reshape(-1)
                        test_caption_words = keywordtrans_srs[test_caption_words_index]
                        test_caption = ' '.join(test_caption_words) 

                    test_caption = ' '.join(test_caption_words)
                    test_caption = test_caption.replace('<bos> ', '')
                    test_caption = test_caption.replace('<eos>', '')
                    test_caption = test_caption.replace(' <eos>', '')
                    test_caption = test_caption.replace('<pad> ', '')
                    test_caption = test_caption.replace(' <pad>', '')
                    test_caption = test_caption.replace(' <unk>', '')
                    test_caption = test_caption.replace('<unk> ', '')

                    if (test_caption == ""):
                        test_caption = '.'

                   
                    test_caption_list.append(test_caption)
             
                    
            df = pd.DataFrame(np.array([test_video_IDs, test_caption_list]).T)
            df.to_csv(output_testset, index=False, header=False)
            
            
            result = {}
            with open(output_testset, 'r') as test_file:
                for line in test_file:
                    line = line.rstrip()
                    test_id, caption = line.split(',')
                    result[test_id] = caption
                    
            bleu= list()
            for item in test_vid_caption:
                score_per_video = list()
                captions = [x.rstrip('.') for x in item['caption']]
                score_per_video.append(BLEU(result[item['id']],captions,True))
                bleu.append(score_per_video[0])
            avg = sum(bleu) / len(bleu)
            print("The Average BLEU Score of model is: " + str(avg))

            if (len(top_BLEU) < num_top_BLEU):
                top_BLEU.append(avg)
                print ("Saving model with BLEU Score : %.4f ..." %(avg))
                model.saver.save(sess, './models/model' + str(avg)[2:6], global_step=epoch)
            else:
                if (avg > min(top_BLEU)):
                    # Remove min. BLEU score.
                    top_BLEU.remove(min(top_BLEU))
                    top_BLEU.append(avg)
                    print ("Saving model with BLEU score: %.4f" %(avg))
                    model.saver.save(sess, './models/model' + str(avg)[2:6], global_step=epoch)
                    
                    
            top_BLEU.sort(reverse=True)
            
            print ("Highest [%d] BLEU scores: " %(num_top_BLEU), ["%.4f" % x for x in top_BLEU])

            print ("Epoch# %d, Loss: %.4f, Average BLEU score: %.4f, Time taken: %.2fs" %(epoch, loss, avg, (time.time() - start_time)))
