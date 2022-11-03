# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Basics
import sys
import os
import threading
try:
	import Queue
except ImportError:
	import queue as Queue
import math
import numpy as np
import argparse as ap
import datetime as dt
import subprocess as sp
import gc
import time
from parse import parse

# logging
# wandb
have_wandb = False
try:
    import wandb
    have_wandb = True
except ImportError:
    pass

# mlperf logger
import utils.mlperf_log_utils as mll

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom
from utils import utils
from utils import losses
from utils import parsing_helpers as ph
from utils import optimizer as uoptim
from utils.spatial_local_sampler import SpatialLocalSampler
from utils.distributed import DistributedSampler
from utils.scheduler import PartialScheduler as Scheduler
from data import cam_hdf5_dataset as cam
from architecture import deeplab_xception

#warmup scheduler
have_warmup_scheduler = False
try:
    from warmup_scheduler import GradualWarmupScheduler
    have_warmup_scheduler = True
except ImportError:
    pass

#vis stuff
from PIL import Image
from utils import visualizer as vizc

#DDP
from mpi4py import MPI
import torch.distributed as dist
try:
    from apex import amp
    import apex.optimizers as aoptim
    from apex.parallel import DistributedDataParallel as DDP
    have_apex = True
except ImportError:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
    have_apex = False

import horovod.torch as hvd

#comm wrapper
from utils import comm


#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def printr(msg, rank=0):
    #if hvd.rank() == rank:
    if hvd.rank() == 0:
        print(msg)

def forward_thread_fn(req, resp, net, criterion, class_weights, fpw_1, fpw_2, distributed_train_sampler, train_set, __none):
    time.sleep(1)

    while True:
        inputs, label, filename = req.get()
        if inputs is None:
            return

        with torch.no_grad():
            outputs = net.forward(inputs)

        loss, loss_per_inp = criterion(outputs, label, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
        for i in range(len(filename)):
            distributed_train_sampler.append_importance(
                train_set.filename2idx(os.path.basename(filename[i])),
                    torch.mean(loss_per_inp[i]).item())

        # Let the main thread know that we are done with this sample
        resp.put((inputs, label, filename, outputs))


#main function
def main(pargs):

    # this should be global
    global have_wandb

    # Determine if using importance samping and LR adjustment
    importance_adjustLR = False
    importance_drop_ratio = 0.0
    if pargs.importance != "disabled":
        for s in pargs.importance.split("+"):
            if s == "adjustLR":
                importance_adjustLR = True
                continue

            r = parse("drop-{:d}perc", s)
            if r:
                importance_drop_ratio = (float(r[0]) / 100)

    if importance_adjustLR and pargs.start_lr > 0:
        pargs.start_lr = pargs.start_lr * (1.0 / (1.0 - importance_drop_ratio))

    hvd.init()
    comm_rank = hvd.rank()
    comm_size = hvd.size()
    comm_local_rank = hvd.local_rank()

    # For backward compatibility of the rest of the code
    pargs.local_rank = comm_local_rank
    pargs.local_size = hvd.local_size()
    #print("rank: {}, size: {}, local_rank: {}".format(comm_rank, comm_size, comm_local_rank))

    MPI.COMM_WORLD.Barrier()

    #set seed
    seed = pargs.seed
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda", pargs.local_rank)
        torch.cuda.manual_seed(seed)
        #necessary for AMP to work
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #print("comm_size: {}, comm_rank: {}, MPI size: {}, MPI rank: {}".format(
    #    comm_size, comm_rank, MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()))
    #sys.exit(0)

    if comm_rank == 0:
        print(pargs)

    if pargs.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # set up logging
    pargs.logging_frequency = max([pargs.logging_frequency, 1])
    log_file = os.path.normpath(os.path.join(pargs.output_dir, "logs", pargs.run_tag + ".log"))
    logger = mll.mlperf_logger(log_file, "deepcam", "Fujitsu", comm_size=comm_size, comm_rank=comm_rank)

    logger.log_start(key = "init_start", sync = True)
    logger.log_event(key = "cache_clear")
    logger.log_event(key = "seed", value = seed)

    #visualize?
    visualize = (pargs.training_visualization_frequency > 0) or (pargs.validation_visualization_frequency > 0)
        
    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    if pargs.stage_dir is not None:
        root_dir = pargs.stage_dir
    if pargs.train_data_dir_prefix == '/':
        train_dir = os.path.join(root_dir, "train")
    else:
        train_dir = os.path.join(pargs.train_data_dir_prefix)
    if pargs.stage_dir is not None:
        train_dir = os.path.join(train_dir, str(comm_rank))

    if pargs.validation_data_dir_prefix == '/':
        validation_dir = os.path.join(root_dir, "validation")
    else:
        validation_dir = os.path.join(pargs.validation_data_dir_prefix)
    if pargs.stage_dir is not None:
        validation_dir = os.path.join(validation_dir, str(comm_rank))


    output_dir = pargs.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    if comm_rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not os.path.isdir(pargs.local_output_dir):
            os.makedirs(pargs.local_output_dir)
        if visualize and not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
    
    # Setup WandB
    if not pargs.enable_wandb:
        have_wandb = False
    if have_wandb and (comm_rank == 0):
        # get wandb api token
        certfile = os.path.join(pargs.wandb_certdir, ".wandbirc")
        try:
            with open(certfile) as f:
                token = f.readlines()[0].replace("\n","").split()
                wblogin = token[0]
                wbtoken = token[1]
        except IOError:
            print("Error, cannot open WandB certificate {}.".format(certfile))
            have_wandb = False

        if have_wandb:
            # log in: that call can be blocking, it should be quick
            sp.call(["wandb", "login", wbtoken])
        
            #init db and get config
            resume_flag = pargs.run_tag if pargs.resume_logging else False
            wandb.init(entity = wblogin, project = 'DeepCAM-ABCI_ImportanceSampling', name = pargs.run_tag, id = pargs.run_tag, resume = resume_flag)
            config = wandb.config
        
            #set general parameters
            config.root_dir = root_dir
            config.train_dir = train_dir
            config.validation_dir = validation_dir
            config.output_dir = pargs.output_dir
            config.max_epochs = pargs.max_epochs
            config.local_batch_size = pargs.local_batch_size
            config.num_workers = comm_size
            config.num_nodes = int(comm_size / pargs.local_size)
            config.channels = pargs.channels
            config.optimizer = pargs.optimizer
            config.start_lr = pargs.start_lr
            config.adam_eps = pargs.adam_eps
            config.weight_decay = pargs.weight_decay
            config.model_prefix = pargs.model_prefix
            config.amp_opt_level = pargs.amp_opt_level
            config.loss_weight_pow = pargs.loss_weight_pow
            config.lr_warmup_steps = pargs.lr_warmup_steps
            config.lr_warmup_factor = pargs.lr_warmup_factor
            config.fraction = pargs.fraction
            config.importance = pargs.importance
            
            # lr schedule if applicable
            if pargs.lr_schedule:
                for key in pargs.lr_schedule:
                    config.update({"lr_schedule_"+key: pargs.lr_schedule[key]}, allow_val_change = True)

    # Define architecture
    n_input_channels = len(pargs.channels)
    n_output_channels = 3
    net = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank)

    if have_apex and torch.cuda.is_available():
        net.cuda()
    else:
        net.to(device)

    net_cpu = deeplab_xception.DeepLabv3_plus(n_input = n_input_channels, 
                                          n_classes = n_output_channels, 
                                          os=16, pretrained=False, 
                                          rank = comm_rank)
    net_cpu.to(torch.device("cpu"))
    
    #select loss
    loss_pow = pargs.loss_weight_pow
    #some magic numbers
    class_weights = [0.986267818390377**loss_pow, 0.0004578708870701058**loss_pow, 0.01327431072255291**loss_pow]
    fpw_1 = 2.61461122397522257612
    fpw_2 = 1.71641974795896018744
    criterion = losses.fp_loss

    #select optimizer
    optimizer = None
    if pargs.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
    elif pargs.optimizer == "LAMB":
        if have_apex:
            optimizer = aoptim.FusedLAMB(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay)
        else:
            optimizer = uoptim.Lamb(net.parameters(), lr = pargs.start_lr, eps = pargs.adam_eps, weight_decay = pargs.weight_decay, clamp_value = torch.iinfo(torch.int32).max)
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(pargs.optimizer))

    # Horovod: (optional) compression algorithm.
    #compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                            named_parameters=net.named_parameters(),
                                            compression=compression)

    if have_apex:
        #wrap model and opt into amp
        net, optimizer = amp.initialize(net, optimizer, opt_level = pargs.amp_opt_level)

    ##make model distributed
    #if have_apex:
    #    #wrap model and opt into amp
    #    if pargs.deterministic:
    #        net = DDP(net, delay_allreduce = True)
    #    else:
    #        net = DDP(net)
    #else:
    #    #torch.nn.parallel.DistributedDataParallel
    #    net = DDP(net, device_ids=[pargs.local_rank], output_device=pargs.local_rank)

    #restart from checkpoint if desired
    #if (comm_rank == 0) and (pargs.checkpoint):
    #load it on all ranks for now
    if pargs.checkpoint:
        checkpoint = torch.load(pargs.checkpoint, map_location = device)
        start_step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
        if have_apex:
            amp.load_state_dict(checkpoint['amp'])
    else:
        start_step = 0
        start_epoch = 0
        
    #select scheduler
    if pargs.lr_schedule:
        scheduler_after = ph.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, last_step = start_step)

        if not have_warmup_scheduler and (pargs.lr_warmup_steps > 0):
            raise ImportError("Error, module {} is not imported".format("warmup_scheduler"))
        elif (pargs.lr_warmup_steps > 0):
            scheduler = GradualWarmupScheduler(optimizer, multiplier=pargs.lr_warmup_factor, total_epoch=pargs.lr_warmup_steps, after_scheduler=scheduler_after)
        else:
            scheduler = scheduler_after
        
    #broadcast model and optimizer state
    #steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False).to(device)
    #dist.broadcast(steptens, src = 0)
    #steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False)
    #steptens = hvd.broadcast(steptens.cpu(), root_rank = 0, name='steptens').item()
    
    #unpack the bcasted tensor
    #start_step = steptens.cpu().numpy()[0]
    #start_epoch = steptens.cpu().numpy()[1]

    # Horovod: broadcast parameters & optimizer state.
    #hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    #hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Logging hyperparameters
    logger.log_event(key = "global_batch_size", value = (pargs.local_batch_size * comm_size))
    logger.log_event(key = "opt_name", value = pargs.optimizer)
    logger.log_event(key = "opt_base_learning_rate", value = pargs.start_lr * pargs.lr_warmup_factor)
    logger.log_event(key = "opt_learning_rate_warmup_steps", value = pargs.lr_warmup_steps)
    logger.log_event(key = "opt_learning_rate_warmup_factor", value = pargs.lr_warmup_factor)
    logger.log_event(key = "opt_epsilon", value = pargs.adam_eps)

    # log events for closed division
    logger.log_event(key = "opt_adam_epsilon", value = pargs.adam_eps)
    logger.log_event(key = "save_frequency", value = pargs.save_frequency)
    logger.log_event(key = "validation_frequency", value = pargs.validation_frequency)
    logger.log_event(key = "logging_frequency", value = pargs.logging_frequency)
    logger.log_event(key = "loss_weight_pow", value = pargs.loss_weight_pow)
    logger.log_event(key = "opt_weight_decay", value = pargs.weight_decay)

    # do sanity check
    if pargs.max_validation_steps is not None:
        logger.log_event(key = "invalid_submission")
    
    #for visualization
    if visualize:
        viz = vizc.CamVisualizer()   
    
    # Train network
    if have_wandb and (comm_rank == 0):
        wandb.watch(net)
    
    step = start_step
    epoch = start_epoch
    eval_epoch = epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else ph.get_lr(scheduler, pargs.lr_schedule)
    net.train()

    # start training
    logger.log_end(key = "init_stop", sync = True)
    logger.log_start(key = "run_start", sync = True)

    MPI.COMM_WORLD.Barrier()
    if hvd.rank() == 0:
        print("Staging input data on all ranks..")

    # run staging
    if pargs.stage_dir is not None:
        logger.log_start(key = "staging_start", sync = True)

        # data_staging.sh takes care of rank inclusion into folder names
        if pargs.debug:
            staging_command='./data_staging.sh {} {} {} {} {} {}'.format(pargs.data_dir_prefix, pargs.stage_dir, comm_rank, comm_size, pargs.local_rank, pargs.debug)
        else:
            staging_command='./data_staging.sh {} {} {} {} {}'.format(pargs.data_dir_prefix, pargs.stage_dir, comm_rank, comm_size, pargs.local_rank)

        staging_ret = sp.run(staging_command, shell=True).returncode
        if staging_ret != 0:
            raise Exception('staging was failed')

        logger.log_end(key = "staging_stop", sync = True)

        local_train_dir = os.path.join(pargs.stage_dir, "train", str(comm_rank))
        #num_local_train_samples_ = torch.Tensor([len([ x for x in os.listdir(local_train_dir) if x.endswith('.h5') ])]).to(device)
        #dist.all_reduce(num_local_train_samples_, op=dist.ReduceOp.MAX)
        #num_local_train_samples = num_local_train_samples_.long().item()

        # XXX: nr. of local train samples is same across all ranks
        num_local_train_samples_ = len([ x for x in os.listdir(local_train_dir) if x.endswith('.h5') ])
        num_local_train_samples = num_local_train_samples_

        local_validation_dir = os.path.join(pargs.stage_dir, "validation", str(comm_rank))
        num_local_validation_samples = len([ x for x in os.listdir(local_validation_dir) if x.endswith('.h5') ])
    else:
        num_local_train_samples = pargs.num_global_train_samples
        num_local_validation_samples = pargs.num_global_validation_samples


    # Only one shard (data is local to each rank)
    num_train_data_shards = 1
    train_dataset_comm_rank = 0

    if pargs.max_inter_threads == 0:
        timeout = 0
    else:
        timeout = 300
    pargs.max_inter_threads = 0
    timeout = 0

    train_allow_uneven_distribution = True
    if pargs.dummy:
        train_allow_uneven_distribution = False

    train_pin_memory = pargs.pin_memory
    train_drop_last = True

    if train_drop_last == False:
        assert(num_local_train_samples % pargs.local_batch_size == 0)

    MPI.COMM_WORLD.Barrier()
    if hvd.rank() == 0:
        print("Setting up DataLoaders on all ranks..")

    # Set up the data feeder
    # train
    train_set = cam.CamDistributedDataset(train_dir,
                               statsfile = os.path.join(pargs.data_dir_prefix, 'stats.h5'),
                               channels = pargs.channels,
                               allow_uneven_distribution = train_allow_uneven_distribution,
                               shuffle = True, 
                               preprocess = True,
                               padding = False,
                               dummy = pargs.dummy,
                               num_local_samples = num_local_train_samples,
                               comm_size = 1,
                               comm_rank = 0,
                               global_size = comm_size,
                               global_rank = comm_rank,
                               seed = pargs.seed,
                               enable_cache = True)

    # Sampling happens locally
    distributed_train_sampler = DistributedSampler(train_set,
                                                   num_replicas = 1,
                                                   rank = 0,
                                                   seed = pargs.seed,
                                                   comm_rank = comm_rank,
                                                   comm_size = comm_size,
                                                   importance_sampling_mode = pargs.importance)

    train_loader = DataLoader(train_set,
                              pargs.local_batch_size,
                              shuffle = (distributed_train_sampler is None),
                              sampler = distributed_train_sampler,
                              num_workers = pargs.max_inter_threads,
                              pin_memory = train_pin_memory,
                              drop_last = train_drop_last,
                              timeout = timeout,
                              #prefetch_factor = 1,
                              #persistent_workers = (pargs.max_inter_threads > 0)
                              )

    train_scheduler = Scheduler(train_set,
                local_batch_size = pargs.local_batch_size,
                fraction = pargs.fraction,
                seed = pargs.seed)
    MPI.COMM_WORLD.Barrier()

    num_validation_data_shards = 1
    validation_dataset_comm_rank = 0

    validation_allow_uneven_distribution = True
    if pargs.dummy:
        validation_allow_uneven_distribution = False

    validation_padding = False
    validation_drop_last = True
    validation_pin_memory = pargs.pin_memory

    # validation: we only want to shuffle the set if we are cutting off validation after a certain number of steps
    validation_set = cam.CamDataset(validation_dir, 
                               statsfile = os.path.join(pargs.data_dir_prefix, 'stats.h5'),
                               channels = pargs.channels,
                               allow_uneven_distribution = validation_allow_uneven_distribution,
                               shuffle = (pargs.max_validation_steps is not None),
                               preprocess = True,
                               padding = validation_padding,
                               dummy = pargs.dummy,
                               num_local_samples = num_local_validation_samples,
                               comm_size = num_validation_data_shards,
                               comm_rank = validation_dataset_comm_rank)

    # use batch size = 1 here to make sure that we do not drop a sample
    validation_loader = DataLoader(validation_set,
                                   1,
                                   shuffle = False,
                                   num_workers = pargs.max_inter_threads,
                                   pin_memory = validation_pin_memory,
                                   drop_last = validation_drop_last,
                                   timeout = timeout,
                                   #prefetch_factor = 1,
                                   #persistent_workers = (pargs.max_inter_threads > 0)
                                   )


    # Loss compute threads
    forward_reqs = Queue.Queue()
    forward_resp = Queue.Queue()
    forward_threads = []
    for i in range(10):
        thread = threading.Thread(
            target = forward_thread_fn,
            args = (forward_reqs, forward_resp, net_cpu, losses.fp_loss_cpu, class_weights, fpw_1, fpw_2, distributed_train_sampler, train_set, None))
        thread.daemon = True
        thread.start()
        forward_threads.append(thread)


    MPI.COMM_WORLD.Barrier()
    if hvd.rank() == 0:
        print("DataLoaders are ready on all ranks.")

    # log size of datasets
    logger.log_event(key = "train_samples", value = pargs.num_global_train_samples)
    if pargs.max_validation_steps is not None:
        logger.log_event(key = "eval_samples", value = 
          min([pargs.num_global_validation_samples, pargs.max_validation_steps * pargs.local_batch_size * comm_size]))
    else:
        logger.log_event(key = "eval_samples", value = pargs.num_global_validation_samples)

    target_accuracy_reached = False

    # Before actual training, do a round of input caching
    nr_inputs = 0
    for inputs, label, filename in train_loader:
        nr_inputs += 1
    logger.log_event(key = "train_mini_batches", value = nr_inputs)

    # training loop
    while True:
        ts_start_epoch = time.perf_counter()

        # start epoch
        MPI.COMM_WORLD.Barrier()
        logger.log_start(key = "epoch_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync=True)
        train_set.next_epoch()
        distributed_train_sampler.set_epoch(epoch)

        torch.cuda.synchronize()
        train_scheduler.scheduling(epoch)
        torch.cuda.synchronize()

        send_requests, recv_requests = None, None

        t_forward_dispatch = 0
        t_iter = 0
        t_trans = 0
        t_forward = 0
        t_loss = 0
        t_importance = 0
        t_backward = 0
        t_update = 0
        t_eval = 0

        ts_start_forward_dispatch = time.perf_counter()
        # epoch loop

		# Async forward eval dispatch
        nr_inputs = 0
        #for inputs, label, filename in train_loader:
        #    inputs_cpu = inputs.detach()
        #    forward_reqs.put(inputs_cpu)
        #    nr_inputs += 1

            #for input_cpu in inputs_cpu:
            #	forward_reqs.put(torch.unsqueeze(input_cpu, dim=0))
            #	nr_inputs += 1

        ts_forward_dispatch = time.perf_counter()

        ts_start = time.perf_counter()

        for inputs, label, filename in train_loader:
  
            MPI.COMM_WORLD.Barrier()
            ts_iter = time.perf_counter()
            loss_was_small = False

            # forward pass
            # Importance sampling?
            if (pargs.importance != "disabled" and not distributed_train_sampler.is_sample_important(
                        train_set.filename2idx(os.path.basename(filename[0])))):
                # Do we do it on the CPU?
                if nr_inputs < 0:
                    inputs_cpu = inputs.detach()
                    forward_reqs.put((inputs_cpu, label, filename))
                    nr_inputs += 1

                    ts_trans = time.perf_counter()

                    # For CPU processed samples we bail out the iteration here
                    step += 1
                    if pargs.lr_schedule:
                        current_lr = ph.get_lr(scheduler, pargs.lr_schedule)
                        scheduler.step()

                    ts_importance = time.perf_counter()

                    t_iter += (ts_iter - ts_start)
                    t_trans += (ts_trans - ts_iter)
                    t_importance += (ts_importance - ts_trans)
                    ts_start = time.perf_counter()
                    continue

                else:
                    # send to device
                    inputs = inputs.to(device)
                    label = label.to(device)
                    ts_trans = time.perf_counter()

                    with torch.no_grad(): 
                        outputs = net.forward(inputs)

            else:
                # send to device
                inputs = inputs.to(device)
                label = label.to(device)
                ts_trans = time.perf_counter()

                outputs = net.forward(inputs)

            ts_forward = time.perf_counter()

            # Compute loss and average across nodes
            loss, loss_per_inp = criterion(outputs, label, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
            if loss < 1:
                loss_was_small = True

            ts_loss = time.perf_counter()

            if pargs.importance != "disabled":
                for i in range(len(filename)):
                    distributed_train_sampler.append_importance(
                        train_set.filename2idx(os.path.basename(filename[i])),
                            torch.mean(loss_per_inp[i].detach()).item())

                # Do not do the rest of the iteration if sample is not important
                if not distributed_train_sampler.is_sample_important(
                        train_set.filename2idx(os.path.basename(filename[0]))):
                    step += 1
                    if pargs.lr_schedule:
                        current_lr = ph.get_lr(scheduler, pargs.lr_schedule)
                        scheduler.step()

                    ts_importance = time.perf_counter()

                    t_iter += (ts_iter - ts_start)
                    t_trans += (ts_trans - ts_iter)
                    t_forward += (ts_forward - ts_trans)
                    t_loss += (ts_loss - ts_forward)
                    t_importance += (ts_importance - ts_loss)
                    ts_start = time.perf_counter()
                    continue

            ts_importance = time.perf_counter()

            # Backprop
            optimizer.zero_grad()
            if have_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            ts_backward = time.perf_counter()

            if send_requests is not None:
                train_scheduler.synchronize(send_requests, recv_requests)

            # Normalize gradients by L2 norm of gradient of the entire model
            if not have_apex and pargs.optimizer == "LAMB":
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()

            torch.cuda.synchronize()
            send_requests, recv_requests = train_scheduler.communicate(step)
            torch.cuda.synchronize()
            
            # step counter
            step += 1
            
            if pargs.lr_schedule:
                current_lr = ph.get_lr(scheduler, pargs.lr_schedule)
                scheduler.step()
            
            ts_update = time.perf_counter()

            #visualize if requested
            if visualize and (step % pargs.training_visualization_frequency == 0) and (comm_rank == 0):
                # Compute predictions
                predictions = torch.max(outputs, 1)[1]
                
                # extract sample id and data tensors
                sample_idx = np.random.randint(low=0, high=label.shape[0])
                plot_input = inputs.detach()[sample_idx, 0,...].cpu().numpy()
                plot_prediction = predictions.detach()[sample_idx,...].cpu().numpy()
                plot_label = label.detach()[sample_idx,...].cpu().numpy()
                
                # create filenames
                outputfile = os.path.basename(filename[sample_idx]).replace("data-", "training-").replace(".h5", ".png")
                outputfile = os.path.join(plot_dir, outputfile)
                
                # plot
                viz.plot(filename[sample_idx], outputfile, plot_input, plot_prediction, plot_label)
                
                #log if requested
                if have_wandb:
                    img = Image.open(outputfile)
                    wandb.log({"train_examples": [wandb.Image(img, caption="Prediction vs. Ground Truth")]}, step = step)
            
            #log if requested
            if (step % pargs.logging_frequency == 0):

                # allreduce for loss
                loss_avg = loss.detach()
                #dist.reduce(loss_avg, dst=0, op=dist.ReduceOp.SUM)
                #loss_avg_train = loss_avg.item() / float(comm_size)
                loss_avg_train = hvd.allreduce(loss_avg.cpu(), name="loss_avg").item()

                # Compute score
                predictions = torch.max(outputs, 1)[1]
                iou = utils.compute_score(predictions, label, device_id=device, num_classes=3)
                iou_avg = iou.detach()
                #dist.reduce(iou_avg, dst=0, op=dist.ReduceOp.SUM)
                #iou_avg_train = iou_avg.item() / float(comm_size)
                iou_avg_train = hvd.allreduce(iou_avg.cpu(), name="iou_avg").item()
                
                logger.log_event(key = "learning_rate", value = current_lr, metadata = {'epoch_num': epoch+1, 'step_num': step})
                logger.log_event(key = "train_accuracy", value = iou_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
                logger.log_event(key = "train_loss", value = loss_avg_train, metadata = {'epoch_num': epoch+1, 'step_num': step})
                
                if have_wandb and (comm_rank == 0):
                    wandb.log({"train_loss": loss_avg_train, "epoch": epoch}, step = step)
                    wandb.log({"train_accuracy": iou_avg_train, "epoch": epoch}, step = step)
                    wandb.log({"learning_rate": current_lr, "epoch": epoch}, step = step)

            # validation step if desired
            #if (step % pargs.validation_frequency == 0):
            # validation in each epoch
            iou_avg_val = 0
            if (eval_epoch != epoch):
                eval_epoch = epoch
                logger.log_start(key = "eval_start", metadata = {'epoch_num': epoch+1})

                #eval
                net.eval()
                
                count_sum_val = torch.Tensor([0.]).to(device)
                loss_sum_val = torch.Tensor([0.]).to(device)
                iou_sum_val = torch.Tensor([0.]).to(device)
                
                # disable gradients
                with torch.no_grad():
                
                    # iterate over validation sample
                    step_val = 0
                    # only print once per eval at most
                    visualized = False
                    for inputs_val, label_val, filename_val in validation_loader:
                        
                        #send to device
                        inputs_val = inputs_val.to(device)
                        label_val = label_val.to(device)
                        
                        # forward pass
                        outputs_val = net.forward(inputs_val)

                        # Compute loss and average across nodes
                        loss_val, losses_val = criterion(outputs_val, label_val, weight=class_weights, fpw_1=fpw_1, fpw_2=fpw_2)
                        loss_sum_val += loss_val
                        
                        #increase counter
                        count_sum_val += 1.
                        
                        # Compute score
                        predictions_val = torch.max(outputs_val, 1)[1]
                        iou_val = utils.compute_score(predictions_val, label_val, device_id=device, num_classes=3)
                        iou_sum_val += iou_val

                        # Visualize
                        if visualize and (step_val % pargs.validation_visualization_frequency == 0) and (not visualized) and (comm_rank == 0):
                            #extract sample id and data tensors
                            sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                            plot_input = inputs_val.detach()[sample_idx, 0,...].cpu().numpy()
                            plot_prediction = predictions_val.detach()[sample_idx,...].cpu().numpy()
                            plot_label = label_val.detach()[sample_idx,...].cpu().numpy()
                            
                            #create filenames
                            outputfile = os.path.basename(filename[sample_idx]).replace("data-", "validation-").replace(".h5", ".png")
                            outputfile = os.path.join(plot_dir, outputfile)
                            
                            #plot
                            viz.plot(filename[sample_idx], outputfile, plot_input, plot_prediction, plot_label)
                            visualized = True
                            
                            #log if requested
                            if have_wandb:
                                img = Image.open(outputfile)
                                wandb.log({"eval_examples": [wandb.Image(img, caption="Prediction vs. Ground Truth")]}, step = step)
                        
                        #increase eval step counter
                        step_val += 1
                        
                        if (pargs.max_validation_steps is not None) and step_val > pargs.max_validation_steps:
                            break
                        
                # average the validation loss
                #dist.all_reduce(count_sum_val, op=dist.ReduceOp.SUM, async_op=False)
                #dist.reduce(loss_sum_val, dst=0, op=dist.ReduceOp.SUM)
                #dist.all_reduce(iou_sum_val, op=dist.ReduceOp.SUM, async_op=False)
                #loss_avg_val = loss_sum_val.item() / count_sum_val.item()
                #iou_avg_val = iou_sum_val.item() / count_sum_val.item()
                count_sum_val = int(hvd.allreduce(count_sum_val.cpu(), name="count_sum_val").item()) * hvd.size()
                loss_sum_val = float(hvd.allreduce(loss_sum_val.cpu(), name="loss_sum_val").item()) * hvd.size()
                iou_sum_val = float(hvd.allreduce(iou_sum_val.cpu(), name="iou_sum_val").item()) * hvd.size()
                loss_avg_val = loss_sum_val / count_sum_val
                iou_avg_val = iou_sum_val / count_sum_val


                # print results
                logger.log_event(key = "eval_accuracy", value = iou_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})
                logger.log_event(key = "eval_loss", value = loss_avg_val, metadata = {'epoch_num': epoch+1, 'step_num': step})

                # log in wandb
                if have_wandb and (comm_rank == 0):
                    wandb.log({"eval_loss": loss_avg_val, "epoch": epoch}, step=step)
                    wandb.log({"eval_accuracy": iou_avg_val, "epoch": epoch}, step=step)

                # set to train
                net.train()

                logger.log_end(key = "eval_stop", metadata = {'epoch_num': epoch+1})

            #save model if desired
            if (pargs.save_frequency > 0) and (step % pargs.save_frequency == 0):
                logger.log_start(key = "save_start", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
                if comm_rank == 0:
                    checkpoint = {
                        'step': step,
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict()
            }
                    if have_apex:
                        checkpoint['amp'] = amp.state_dict()
                    torch.save(checkpoint, os.path.join(pargs.local_output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )
                logger.log_end(key = "save_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)

            if (step % pargs.validation_frequency == 0) and (iou_avg_val >= pargs.target_iou):
                logger.log_event(key = "target_accuracy_reached", value = pargs.target_iou, metadata = {'epoch_num': epoch+1, 'step_num': step})
                target_accuracy_reached = True
                break;
            
            ts_eval = time.perf_counter()

            t_iter += (ts_iter - ts_start)
            t_trans += (ts_trans - ts_iter)
            t_forward += (ts_forward - ts_trans)
            t_loss += (ts_loss - ts_forward)
            t_importance += (ts_importance - ts_loss)
            t_backward += (ts_backward - ts_importance)
            t_update += (ts_update - ts_backward)
            t_eval += (ts_eval - ts_update)


            ts_start = time.perf_counter()

        if send_requests is not None:
            train_scheduler.synchronize(send_requests, recv_requests)

        torch.cuda.synchronize()
        train_scheduler.clean_local_storage()
        torch.cuda.synchronize()

        # Get loss compute threads' results
        for i in range(nr_inputs):
            inputs, label, filename, outputs = forward_resp.get()
            '''
            loss, loss_per_inp = criterion(outputs, label, weight=class_weights,
                                            fpw_1=fpw_1, fpw_2=fpw_2)
            for i in range(len(filename)):
                distributed_train_sampler.append_importance(
                    train_set.filename2idx(os.path.basename(filename[i])),
                        torch.mean(loss_per_inp[i]).item())
            '''

        if comm_rank == 0:
            print("Got {} loss evaluation results..".format(nr_inputs))
        
        ts_sync_threads = time.perf_counter()
        t_sync_threads = ts_sync_threads - ts_start

        net_cpu.load_state_dict(net.state_dict())
        ts_update_cpu_net = time.perf_counter()
        t_update_cpu_net = ts_update_cpu_net - ts_sync_threads

        t_forward_dispatch = ts_forward_dispatch - ts_start_forward_dispatch

        ts_end_epoch = time.perf_counter()
        # log the epoch
        logger.log_event(key = "t_forward_dispatch", value = t_forward_dispatch, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_iter", value = t_iter, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_trans", value = t_trans, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_forward", value = t_forward, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_loss", value = t_loss, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_importance", value = t_importance, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_backward", value = t_backward, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_update", value = t_update, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_eval", value = t_eval, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_eval", value = t_eval, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_sync_threads", value = t_sync_threads, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "t_update_cpu_net", value = t_update_cpu_net, metadata = {'epoch_num': epoch+1})
        logger.log_event(key = "epoch_duration", value = (ts_end_epoch - ts_start_epoch), metadata = {'epoch_num': epoch+1, 'step_num': step})
        logger.log_end(key = "epoch_stop", metadata = {'epoch_num': epoch+1, 'step_num': step}, sync = True)
        epoch += 1

        # are we done?
        if epoch >= pargs.max_epochs or target_accuracy_reached:
            break

    # run done
    logger.log_end(key = "run_stop", sync = True, metadata = {'status' : 'success'})


if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--wireup_method", type=str, default="nccl-openmpi", choices=["nccl-openmpi", "nccl-slurm", "nccl-slurm-pmi", "mpi"], help="Specify what is used for wiring up the ranks")
    AP.add_argument("--wandb_certdir", type=str, default="/opt/certs", help="Directory in which to find the certificate for wandb logging.")
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--stage_dir", type=str, default=None, help="stage dir from data_dir_prefix")
    AP.add_argument("--train_data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--validation_data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--max_validation_steps", type=int, default=None, help="Number of validation steps to perform. Helps when validation takes a long time. WARNING: setting this argument invalidates submission. It should only be used for exploration, the final submission needs to have it disabled.")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--training_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--validation_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during validation")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "LAMB"], help="Optimizer to use (LAMB requires APEX support).")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_weight_pow", type=float, default=-0.125, help="Decay factor to adjust the weights")
    AP.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for linear LR warmup")
    AP.add_argument("--lr_warmup_factor", type=float, default=1., help="Multiplier for linear LR warmup")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--target_iou", type=float, default=0.82, help="Target IoU score.")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    AP.add_argument("--enable_wandb", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    AP.add_argument("--local_output_dir", type=str)
    AP.add_argument("--shuffle_after_epoch", action='store_true')
    AP.add_argument("--num_global_train_samples", default=121266, type=int)
    AP.add_argument("--num_global_validation_samples", default=15159, type=int)
    AP.add_argument("--num_train_data_shards", default=1, type=int)
    AP.add_argument("--num_validation_data_shards", default=1, type=int)
    AP.add_argument("--pin_memory", action='store_true')
    AP.add_argument("--deterministic", action='store_true')
    AP.add_argument("--seed", default=333, type=int)
    AP.add_argument("--dummy", action='store_true')
    AP.add_argument("--debug", action='store_true')
    AP.add_argument("--num_shuffle_nodes", default=0, type=int)
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
    AP.add_argument("--local_rank", default=0, type=int)
    AP.add_argument("--local_size", default=1, type=int)
    AP.add_argument('--fraction', type=float, default=0.0, help=' Fraction of non-local samples in SpatialLocalSampler. In range of [0,1]. If ratio = 0, it uses local sampling, 1 is global sampling. Default value = 0')
    AP.add_argument('--importance', type=str, default="disabled", help=' Importance sampling mode (default: disabled)')
    pargs = AP.parse_args()
    
    #run the stuff
    main(pargs)
