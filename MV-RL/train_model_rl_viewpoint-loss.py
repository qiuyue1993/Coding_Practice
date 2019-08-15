#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import ipdb as pdb
import json
import random
import shutil
from termcolor import colored
import time

import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

import vr.utils as utils
import vr.preprocess
from vr.data import ClevrDataset, ClevrDataLoader
##### Modified #####
from vr.models import ModuleNet, Seq2Seq, LstmModel, CnnLstmModel, CnnLstmSaModel, Scene2Action
from vr.models import FiLMedNet
from vr.models import FiLMGen

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser()

# Input data
##### Modified #####
parser.add_argument('--train_question_h5', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/questions_vocab/Multi-view-CLEVR-v1_12views/train_questions.h5')
parser.add_argument('--train_features_h5', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/image_features/Multi-view-CLEVR-v1_12views/133/train_features.h5')
parser.add_argument('--train_features_h5_aux', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/image_features/Multi-view-CLEVR-v1_12views/133/train_features_aux.h5')

parser.add_argument('--val_question_h5', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/questions_vocab/Multi-view-CLEVR-v1_12views/val_questions.h5')
parser.add_argument('--val_features_h5', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/image_features/Multi-view-CLEVR-v1_12views/133/val_features.h5')
parser.add_argument('--val_features_h5_aux', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/image_features/Multi-view-CLEVR-v1_12views/133/val_features_aux.h5')

parser.add_argument('--feature_dim', default='256,16,16')
parser.add_argument('--vocab_json', default='/home/qiu/VQA/CLEVR/clevr-iep-master/data/questions_vocab/Multi-view-CLEVR-v1_12views/vocab.json')

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=None, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
  choices=['FiLM', 'PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net / FiLMedNet options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# FiLM only options
parser.add_argument('--set_execution_engine_eval', default=0, type=int)
parser.add_argument('--program_generator_parameter_efficient', default=1, type=int)
parser.add_argument('--rnn_output_batchnorm', default=0, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--encoder_type', default='gru', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--decoder_type', default='linear', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--gamma_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp'])
parser.add_argument('--gamma_baseline', default=1, type=float)
parser.add_argument('--num_modules', default=4, type=int)
parser.add_argument('--module_stem_kernel_size', default=3, type=int)
parser.add_argument('--module_stem_stride', default=1, type=int)
parser.add_argument('--module_stem_padding', default=None, type=int)
parser.add_argument('--module_num_layers', default=1, type=int)  # Only mnl=1 currently implemented
parser.add_argument('--module_batchnorm_affine', default=0, type=int)  # 1 overrides other factors
parser.add_argument('--module_dropout', default=5e-2, type=float)
parser.add_argument('--module_input_proj', default=1, type=int)  # Inp conv kernel size (0 for None)
parser.add_argument('--module_kernel_size', default=3, type=int)
parser.add_argument('--condition_method', default='bn-film', type=str,
  choices=['block-input-film', 'block-output-film', 'bn-film', 'concat', 'conv-film', 'relu-film'])
parser.add_argument('--condition_pattern', default='', type=str)  # List of 0/1's (len = # FiLMs)
parser.add_argument('--use_gamma', default=1, type=int)
parser.add_argument('--use_beta', default=1, type=int)
parser.add_argument('--use_coords', default=1, type=int)  # 0: none, 1: low usage, 2: high usage
parser.add_argument('--grad_clip', default=0, type=float)  # <= 0 for no grad clipping
parser.add_argument('--debug_every', default=float('inf'), type=float)  # inf for no pdb
parser.add_argument('--print_verbose_every', default=float('inf'), type=float)  # inf for min print

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
  choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
  choices=['maxpool2', 'maxpool3', 'maxpool4', 'maxpool5', 'maxpool7', 'maxpoolfull', 'none',
           'avgpool2', 'avgpool3', 'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=600000, type=int)
parser.add_argument('--optimizer', default='Adam',
  choices=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD'])
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--avoid_checkpoint_override', default=0, type=int)
parser.add_argument('--record_loss_every', default=1, type=int)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--time', default=0, type=int)
parser.add_argument('--device_idd', default=0, type=int)

##### Modified #####
# Scene2Action options
parser.add_argument('--scene2action_feat_dim', default=[256,16,16])
parser.add_argument('--scene2action_hidden_dim', default=1024, type=int)
parser.add_argument('--scene2action_action_dim', default=13, type=int)
parser.add_argument('--scene2action_dropout', default=0, type=float)
parser.add_argument('--scene2action_word_vocab_size', default=100, type=int)
parser.add_argument('--scene2action_word_embed_size', default=200, type=int)
parser.add_argument('--scene2action_lstm_hidden_size', default=4096, type=int)
parser.add_argument('--scene2action_lstm_num_layers', default=1, type=int)
parser.add_argument('--scene2action_gamma', default=0.95, type=float)

# Hyper-parameters for policy gradients
parser.add_argument('--maximum_action_number', default=3, type=int) 
parser.add_argument('--end_action_index', default=12, type=int) 

# Tensorboard related information
parser.add_argument('--root_log_dir', default='/', type=str)
parser.add_argument('--log_dir', default='/', type=str)
parser.add_argument('--log_interval', default=10, type=int)
parser.add_argument('--save_interval', default=100, type=int)

# Hyperparameter for route loss
parser.add_argument('--route_loss_rate', default=1.0, type=float)

##### Modified #####
## A function for decay and normalize reward of Policy Gradients
def decay_normalize_loss(Minus_reward_currentturn, gamma):
  temp_reward_currentturn = torch.zeros_like(Minus_reward_currentturn)
  for index in range(len(temp_reward_currentturn)):
    temp_reward_currentturn[index] = (-1.0)*Minus_reward_currentturn[index] ##### Is this OK?! #####
  
  discounted_rewards = torch.zeros_like(temp_reward_currentturn)

  running_add = 0

  for t in reversed(range(0, len(temp_reward_currentturn))):
    running_add = running_add*gamma + temp_reward_currentturn[t]
    discounted_rewards[t] = running_add

  if(len(discounted_rewards) > 1):
    discounted_rewards -= torch.mean(discounted_rewards, dim=0)
 
    if (discounted_rewards.std() != 0):
      discounted_rewards /= discounted_rewards.std()

  return discounted_rewards

def main(args):
  torch.autograd.set_detect_anomaly(True)
  # for reproducibility
  torch.cuda.set_device(args.device_idd)
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  torch.backends.cudnn.benckmark = False
  torch.backends.cudnn.deterministic = True

  if args.randomize_checkpoint_path == 1:
    name, ext = os.path.splitext(args.checkpoint_path)
    num = random.randint(1, 1000000)
    args.checkpoint_path = '%s_%06d%s' % (name, num, ext)
  print('Will save checkpoints to %s' % args.checkpoint_path)

  vocab = utils.load_vocab(args.vocab_json)


  ##### Modified #####
  ### For Tensorboard ###
  #log_interval_num = args.log_interval
  #save_interval_num = args.save_interval
  log_dir = os.path.join(args.root_log_dir, args.log_dir)
  os.mkdir(log_dir)
  os.mkdir(os.path.join(log_dir, 'runs'))

  writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))


  ##### Modified #####
  if args.use_local_copies == 1:
    shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
    shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
    shutil.copy(args.train_features_h5_aux, '/tmp/train_features_aux.h5')

    shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
    shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
    shutil.copy(args.val_features_h5_aux, '/tmp/val_features_aux.h5')

    args.train_question_h5 = '/tmp/train_questions.h5'
    args.train_features_h5 = '/tmp/train_features.h5'
    args.train_features_h5_aux = '/tmp/train_features_aux.h5'

    args.val_question_h5 = '/tmp/val_questions.h5'
    args.val_features_h5 = '/tmp/val_features.h5'
    args.val_features_h5_aux = '/tmp/val_features_aux.h5'


  question_families = None
  if args.family_split_file is not None:
    with open(args.family_split_file, 'r') as f:
      question_families = json.load(f)

  ##### Modified #####
  train_loader_kwargs = {
    'question_h5': args.train_question_h5,
    'feature_h5': args.train_features_h5,
    'feature_h5_aux': args.train_features_h5_aux,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle_train_data == 1,
    'question_families': question_families,
    'max_samples': args.num_train_samples,
    'num_workers': args.loader_num_workers,
  }
  val_loader_kwargs = {
    'question_h5': args.val_question_h5,
    'feature_h5': args.val_features_h5,
    'feature_h5_aux': args.val_features_h5_aux,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'question_families': question_families,
    'max_samples': args.num_val_samples,
    'num_workers': args.loader_num_workers,
  }

  with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
       ClevrDataLoader(**val_loader_kwargs) as val_loader:
    train_loop(args, train_loader, val_loader, writer)

  ##### Modified #####
  if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
    os.remove('/tmp/train_questions.h5')
    os.remove('/tmp/train_features.h5')
    os.remove('/tmp/train_features_aux.h5')
    os.remove('/tmp/val_questions.h5')
    os.remove('/tmp/val_features.h5')
    os.remove('/tmp/val_features_aux.h5')


def train_loop(args, train_loader, val_loader, writer):
  vocab = utils.load_vocab(args.vocab_json)
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  pg_best_state, ee_best_state, baseline_best_state = None, None, None

  # Set up model
  optim_method = getattr(torch.optim, args.optimizer)

  ##### Modified #####
  scene2action_model, scene2action_optimizer = None, None


  s2a_kwargs = {
    'feat_dim': args.scene2action_feat_dim,
    'hidden_dim': args.scene2action_hidden_dim,
    'action_dim': args.scene2action_action_dim, 
    'dropout': args.scene2action_dropout,
    'word_vocab_size': args.scene2action_word_vocab_size,
    'word_embed_size': args.scene2action_word_embed_size,
    'lstm_hidden_size': args.scene2action_lstm_hidden_size,
    'lstm_num_layers': args.scene2action_lstm_num_layers,
  }

  scene2action_model = Scene2Action(**s2a_kwargs)
  scene2action_model.cuda()
  scene2action_model.train()
  scene2action_optimizer = optim_method(scene2action_model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
  
  print("Here is the scene to action network: ")
  print(scene2action_model)


  if args.model_type in ['FiLM', 'PG', 'PG+EE']:
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = optim_method(program_generator.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    print('Here is the conditioning network:')
    print(program_generator)
  if args.model_type in ['FiLM', 'EE', 'PG+EE']:
    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = optim_method(execution_engine.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    print('Here is the conditioned network:')
    print(execution_engine)
  if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    baseline_optimizer = optim_method(params,
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type

  loss_fn = torch.nn.CrossEntropyLoss().cuda()

  ##### Modified #####
  loss_s2a = torch.nn.CrossEntropyLoss().cuda()

  ##### Modified #####
  stats = {
    'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
    'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
    'best_val_acc': -1, 'model_t': 0,
    'train_losses_s2a': [],
  }
  t, epoch, reward_moving_average = 0, 0, 0

  ##### Modified #####
  set_mode('train', [scene2action_model, program_generator, execution_engine, baseline_model])

  print('train_loader has %d samples' % len(train_loader.dataset))
  print('val_loader has %d samples' % len(val_loader.dataset))

  num_checkpoints = 0
  epoch_start_time = 0.0
  epoch_total_time = 0.0
  train_pass_total_time = 0.0
  val_pass_total_time = 0.0
  running_loss = 0.0

  while t < args.num_iterations:
    if (epoch > 0) and (args.time == 1):
      epoch_time = time.time() - epoch_start_time
      epoch_total_time += epoch_time
      print(colored('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white'))
      print(colored('Epoch Pass Time      : ' + str(epoch_time), 'white'))
    epoch_start_time = time.time()

    epoch += 1

    print('Starting epoch %d' % epoch)

    for batch in train_loader:

      t += 1
      print("Current batch " + str(t)) #------------------------------------------------------------------------------------#

      ##### Modified #####  
      questions, _, feats, feats_aux, answers, programs, _ = batch
      if isinstance(questions, list):
        questions = questions[0]
      questions_var = Variable(questions.cuda())

      feats_var = Variable(feats.cuda())
      ##### Modified ####
      feats_var_aux = Variable(feats_aux.cuda())
      answers_var = Variable(answers.cuda())

      #print("answers var 0 " + str(answers_var.size(0))) ### 64

      if programs[0] is not None:
        programs_var = Variable(programs.cuda())

      reward = None

      ##### Modified #####---------------------------------------------------------
      ### Preprocess batch started ###      

      # For each data in the current batch (step-by-step)


      avg_action_length = 0.0
      total_repeated_rate = 0.0
      count_turn = 0

      for turn in range(feats_var.size(0)):
        #print("Current turn " + str(turn)) #------------------------------------------------------------------------------------#

        set_mode('eval', [program_generator, execution_engine])
        current_action = -1
        current_count = 0
        current_scene = feats_var[turn]
        current_question = questions_var[turn]
        current_answer = answers_var[turn]
        actions = []
        flag = 0
      
        while ((current_action != args.end_action_index) and (current_count < args.maximum_action_number)):
          current_count += 1
          current_action, action_propability = scene2action_model(current_scene, current_question)
          actions.append(current_action.item())
          #print("Current_action " + str(current_action))
          #print("action_propability " + str(action_propability))

          #print("current action after " + str(current_action))
          if (current_action != args.end_action_index):
            if (current_action == 0):
              current_scene = current_scene + feats_var[turn]
            else:
              current_scene = current_scene + feats_var_aux[turn][current_action-1] ##### To check #####

          temp_question = current_question.view(1,-1).clone() ##### To check #####
          programs_pred = program_generator(temp_question)
          current_scene = current_scene.view(1,256,16,16) ##### To check #####
          scores = execution_engine(current_scene, programs_pred)

          current_answer = current_answer.view(-1) ##### To check #####
          loss_current = loss_fn(scores, current_answer)


          if (flag == 0):
            Minus_reward_currentturn = torch.zeros(1,1)
            Minus_reward_currentturn[0][0] = torch.from_numpy(np.array(loss_current.data.cpu().numpy()))
          else:
            temp = torch.zeros(1,1)
            temp[0][0] = torch.from_numpy(np.array(loss_current.data.cpu().numpy()))
            Minus_reward_currentturn = torch.cat((Minus_reward_currentturn, temp), 0) ### ??? ###

          if (flag == 0):
            Logits_currentturn = torch.zeros(1,1)
            Logits_currentturn[0][0] = torch.from_numpy(np.array(action_propability.data.cpu().numpy()))
            flag = flag + 1
          else:
            temp = torch.zeros(1,1)
            temp[0][0] = torch.from_numpy(np.array(action_propability.data.cpu().numpy()))

            Logits_currentturn = torch.cat((Logits_currentturn, temp), 0) ##### To check #####


        #print("Current Action " + str(current_action)) #------------------------------------------------------------------------------------#


        vt = decay_normalize_loss(Minus_reward_currentturn, args.scene2action_gamma)
        Logits_currentturn = Logits_currentturn.cuda()

        vt = vt.cuda()

        current_route_length = torch.tensor(len(actions))
        current_route_length_cuda = current_route_length.cuda()
        ##### Modified #####
        ### Add route loss here ### route_loss_rate
        loss_current_s2a = torch.mean(Logits_currentturn*vt) + args.route_loss_rate*current_route_length_cuda
        loss_current_s2a.requires_grad=True

        scene2action_optimizer.zero_grad()
        
        loss_current_s2a = loss_current_s2a.cuda()
        loss_current_s2a.backward()
        scene2action_optimizer.step()

        feats_var[turn] = current_scene

        avg_action_length = avg_action_length + len(actions)

        repeated_actions = 0
        for a1 in range(len(actions) - 1):
          for a2 in range(a1 + 1, len(actions)):
             if actions[a1] == actions[a2]:
               repeated_actions = repeated_actions + 1
               break
        total_repeated_rate = total_repeated_rate + (repeated_actions)/(float)(len(actions))
        count_turn = count_turn + 1

      writer.add_scalar('scene2action_loss',loss_current_s2a.item(),t)
      
      ### Add avg length, no_repeated rate
      writer.add_scalar('avg_route_length', avg_action_length/(float)(count_turn), t)
      writer.add_scalar('avg_repeat_rate', total_repeated_rate/(float)(count_turn), t)  

      ### Turn actions to image ###
      image_ = torch.zeros(3,13,13)
      print("actions ----------------------- ")
      print(actions)


      for ii_ in range(len(actions)):
        image_[0][actions[ii_]][ii_] = 0
        image_[1][actions[ii_]][ii_] = 1
        image_[2][actions[ii_]][ii_] = 1

      writer.add_image('selected_viewpoints',image_,t)
          
      set_mode('train', [scene2action_model, program_generator, execution_engine, baseline_model])
      #exit()
      ### Preprocess batch ended ###
      ##### Modified #####---------------------------------------------------------

      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(questions_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type == 'EE':
        # Train execution engine with ground-truth programs
        ee_optimizer.zero_grad()
        scores = execution_engine(feats_var, programs_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        ee_optimizer.step()
      elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(questions_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)
        raw_reward = (preds == answers).float()
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          program_generator.reinforce_backward(centered_reward.cuda())
          pg_optimizer.step()
      elif args.model_type == 'FiLM':
        if args.set_execution_engine_eval == 1:
          set_mode('eval', [execution_engine])
        programs_pred = program_generator(questions_var)
        scores = execution_engine(feats_var, programs_pred)
        loss = loss_fn(scores, answers_var)

        pg_optimizer.zero_grad()
        ee_optimizer.zero_grad()
        if args.debug_every <= -2:
          pdb.set_trace()
        loss.backward()
        if args.debug_every < float('inf'):
          check_grad_num_nans(execution_engine, 'FiLMedNet')
          check_grad_num_nans(program_generator, 'FiLMGen')

        if args.train_program_generator == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
          pg_optimizer.step()
        if args.train_execution_engine == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
          ee_optimizer.step()

      if t % args.record_loss_every == 0:
        running_loss += loss.data.item()
        avg_loss = running_loss / args.record_loss_every
        print(t, avg_loss)
        stats['train_losses'].append(avg_loss)
        stats['train_losses_ts'].append(t)
        ##### Modified #####
        stats['train_losses_s2a'].append(loss_current_s2a)
        if reward is not None:
          stats['train_rewards'].append(reward)
        running_loss = 0.0
      else:
        running_loss += loss.data.item()


      writer.add_scalar('train_losses',loss.data.item(),t)
      if t % args.checkpoint_every == 0:
        num_checkpoints += 1
        print('Checking training accuracy ... ')
        start = time.time()
        """
        ##### Modified #####
        train_acc = check_accuracy(args, scene2action_model, program_generator, execution_engine,
                                   baseline_model, train_loader)
        if args.time == 1:
          train_pass_time = (time.time() - start)
          train_pass_total_time += train_pass_time
          print(colored('TRAIN PASS AVG TIME: ' + str(train_pass_total_time / num_checkpoints), 'red'))
          print(colored('Train Pass Time      : ' + str(train_pass_time), 'red'))
        print('train accuracy is', train_acc)
        print('Checking validation accuracy ...')
        start = time.time()
        ##### Modified #####
        """
        val_acc = check_accuracy(args, scene2action_model, program_generator, execution_engine,
                                 baseline_model, val_loader)
        if args.time == 1:
          val_pass_time = (time.time() - start)
          val_pass_total_time += val_pass_time
          print(colored('VAL PASS AVG TIME:   ' + str(val_pass_total_time / num_checkpoints), 'cyan'))
          print(colored('Val Pass Time        : ' + str(val_pass_time), 'cyan'))
        print('val accuracy is ', val_acc)
        
        ### Val_acc
        writer.add_scalar('val_acc',val_acc, t)


        """
        stats['train_accs'].append(train_acc)
        stats['val_accs'].append(val_acc)
        stats['val_accs_ts'].append(t)

        #if val_acc > stats['best_val_acc']:
        stats['best_val_acc'] = val_acc
        stats['model_t'] = t
        """
        ##### Modified #####
        best_scene2action_state = get_state(scene2action_model)
        best_pg_state = get_state(program_generator)
        best_ee_state = get_state(execution_engine)
        best_baseline_state = get_state(baseline_model)

        ##### Modified #####
        checkpoint = {
          'args': args.__dict__,
          'scene2action_kwargs':s2a_kwargs,
          'scene2action_state':best_scene2action_state,
          'program_generator_kwargs': pg_kwargs,
          'program_generator_state': best_pg_state,
          'execution_engine_kwargs': ee_kwargs,
          'execution_engine_state': best_ee_state,
          'baseline_kwargs': baseline_kwargs,
          'baseline_state': best_baseline_state,
          'baseline_type': baseline_type,
          'vocab': vocab
        }
        for k, v in stats.items():
          checkpoint[k] = v
        print('Saving checkpoint to %s' % args.checkpoint_path)
        torch.save(checkpoint, args.checkpoint_path + '_' + str(t) + '.pt')
        ##### Modified #####
        del checkpoint['scene2action_state']
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        del checkpoint['baseline_state']
        #with open(args.checkpoint_path + '.json', 'w') as f:
        #  json.dump(checkpoint, f)

      if t == args.num_iterations:
        break


def parse_int_list(s):
  if s == '': return ()
  return tuple(int(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = utils.load_program_generator(
      args.program_generator_start_from, model_type=args.model_type)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['question_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['question_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
  else:
    kwargs = {
      'encoder_vocab_size': len(vocab['question_token_to_idx']),
      'decoder_vocab_size': len(vocab['program_token_to_idx']),
      'wordvec_dim': args.rnn_wordvec_dim,
      'hidden_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
    }
    if args.model_type == 'FiLM':
      kwargs['parameter_efficient'] = args.program_generator_parameter_efficient == 1
      kwargs['output_batchnorm'] = args.rnn_output_batchnorm == 1
      kwargs['bidirectional'] = args.bidirectional == 1
      kwargs['encoder_type'] = args.encoder_type
      kwargs['decoder_type'] = args.decoder_type
      kwargs['gamma_option'] = args.gamma_option
      kwargs['gamma_baseline'] = args.gamma_baseline
      kwargs['num_modules'] = args.num_modules
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_dim'] = args.module_dim
      kwargs['debug_every'] = args.debug_every
      pg = FiLMGen(**kwargs)
    else:
      pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = utils.load_execution_engine(
      args.execution_engine_start_from, model_type=args.model_type)
  else:
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_num_layers': args.module_stem_num_layers,
      'module_dim': args.module_dim,
      'module_residual': args.module_residual == 1,
      'module_batchnorm': args.module_batchnorm == 1,
      'classifier_proj_dim': args.classifier_proj_dim,
      'classifier_downsample': args.classifier_downsample,
      'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
      'classifier_batchnorm': args.classifier_batchnorm == 1,
      'classifier_dropout': args.classifier_dropout,
    }
    if args.model_type == 'FiLM':
      kwargs['num_modules'] = args.num_modules
      kwargs['stem_kernel_size'] = args.module_stem_kernel_size
      kwargs['stem_stride'] = args.module_stem_stride
      kwargs['stem_padding'] = args.module_stem_padding
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
      kwargs['module_dropout'] = args.module_dropout
      kwargs['module_input_proj'] = args.module_input_proj
      kwargs['module_kernel_size'] = args.module_kernel_size
      kwargs['use_gamma'] = args.use_gamma == 1
      kwargs['use_beta'] = args.use_beta == 1
      kwargs['use_coords'] = args.use_coords
      kwargs['debug_every'] = args.debug_every
      kwargs['print_verbose_every'] = args.print_verbose_every
      kwargs['condition_method'] = args.condition_method
      kwargs['condition_pattern'] = parse_int_list(args.condition_pattern)
      ee = FiLMedNet(**kwargs)
    else:
      ee = ModuleNet(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs


def get_baseline_model(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = utils.load_baseline(args.baseline_start_from)
  elif args.model_type == 'LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'cnn_num_res_blocks': args.cnn_num_res_blocks,
      'cnn_res_block_dim': args.cnn_res_block_dim,
      'cnn_proj_dim': args.cnn_proj_dim,
      'cnn_pooling': args.cnn_pooling,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'stacked_attn_dim': args.stacked_attn_dim,
      'num_stacked_attn': args.num_stacked_attn,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmSaModel(**kwargs)
  if model.rnn.token_to_idx != vocab['question_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['question_token_to_idx']
      assert vocab['question_token_to_idx'][k] == v
    for token, idx in vocab['question_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['question_token_to_idx'])
  model.cuda()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()


def check_accuracy(args, scene2action_model, program_generator, execution_engine, baseline_model, loader):
  ##### Modified #####
  set_mode('eval', [scene2action_model, program_generator, execution_engine, baseline_model])
  num_correct, num_samples = 0, 0
  for batch in loader:
    ##### Modified #####
    questions, _, feats, feats_aux, answers, programs, _ = batch

    if isinstance(questions, list):
      questions = questions[0]

    questions_var = Variable(questions.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    ##### Modified #####
    feats_var_aux = Variable(feats_aux.cuda(), volatile=True)
    answers_var = Variable(feats.cuda(), volatile=True)
    if programs[0] is not None:
      programs_var = Variable(programs.cuda(), volatile=True)

    ##### Modified #####------------------------------------------
    ### Preprocess batch started ###
    
    # For each data in the current batch (step-by-step)
    for turn in range(feats_var.size(0)):
      current_action = -1
      current_count = 0
      current_scene = feats_var[turn]
      current_question = questions_var[turn]

      while((current_action != args.end_action_index) and (current_count < args.maximum_action_number)):
        current_count = current_count + 1
        
        current_action, _ = scene2action_model(current_scene, current_question)
        if(current_action != args.end_action_index):
          if (current_action == 0):
            current_scene = current_scene + feats_var[turn]
          else:
            current_scene = current_scene + feats_var_aux[turn][current_action-1]

      ### Prepare data for VQA model ###
      feats_var[turn] = current_scene
    ### Preprocess batch ended ###
    ##### Modified #####------------------------------------------

    scores = None  # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = utils.load_vocab(args.vocab_json)
      for i in range(questions.size(0)):
        program_pred = program_generator.sample(Variable(questions[i:i+1].cuda(), volatile=True))
        program_pred_str = vr.preprocess.decode(program_pred, vocab['program_idx_to_token'])
        program_str = vr.preprocess.decode(programs[i], vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
      scores = execution_engine(feats_var, programs_var)
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
                          questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type == 'FiLM':
      programs_pred = program_generator(questions_var)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(questions_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if args.num_val_samples is not None and num_samples >= args.num_val_samples:
      break

  set_mode('train', [scene2action_model, program_generator, execution_engine, baseline_model])
  acc = float(num_correct) / num_samples
  return acc

def check_grad_num_nans(model, model_name='model'):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    num_nans = [np.sum(np.isnan(grad.data.cpu().numpy())) for grad in grads]
    nan_checks = [num_nan == 0 for num_nan in num_nans]
    if False in nan_checks:
      print('Nans in ' + model_name + ' gradient!')
      print(num_nans)
      pdb.set_trace()
      raise(Exception)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
