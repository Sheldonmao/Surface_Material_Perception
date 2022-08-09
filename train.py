"""Specific traning script for blender rendered IR images
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer, VisualizerCapture
from utils.criteria import Criteria
from tqdm import tqdm

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    train_dataset = create_dataset(opt,opt.train_csv)  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    if opt.train_csv == opt.val_csv:        # a trick to prevent multiple preload
        val_dataset = train_dataset
    else:
        val_dataset = create_dataset(opt,opt.val_csv)
    print('The number of training images = %d, validation images = %d' % (train_dataset_size,len(val_dataset)))

    # create a visualizer that display/save images and plots
    if opt.dataset_mode.lower() == "patchblender":
        visualizer = Visualizer(opt)
    elif opt.dataset_mode.lower() == "patchcapture":
        visualizer = VisualizerCapture(opt)   
    else:
        raise(NotImplementedError('not implimented for dataset: {}'.format(opt.dataset_mode)))
        
    train_criteria = Criteria('train')
    total_iters = 0                # the total number of training iterations
    prev_display_time = time.time()

    for epoch in range(opt.load_epoch, opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <load_epoch>, <load_epoch>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        

        for i, data in tqdm(enumerate(train_dataset)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.display_freq == 0:
                t_data = iter_start_time - iter_data_time

            # torch.save(data,os.path.join(opt.outf,'input_data.pt'))
            # print('input_saved')
            # break

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            train_criteria.update(model.get_outputs(),model.get_target())

            if total_iters % opt.display_freq == 0:   # display losses on tensorboard
                losses = model.get_current_losses()
                accuracies = train_criteria.get_avgs()   # retrieve accuracy
                train_criteria.reset()              # after retrieving the average accuracy, clear the accumulator
                results = {**losses, **accuracies}  # merge two dictionaries
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                log_idx = (epoch * train_dataset_size + epoch_iter)/opt.batch_size
                t_display = time.time() - prev_display_time
                prev_display_time = time.time()
                visualizer.print_current_losses(results, epoch, epoch_iter, t_comp, t_data,t_display)
                visualizer.plot_current_losses(results, log_idx)

            if total_iters % (opt.display_freq*10) == 0:   # display images on tensorboard
                model.compute_visuals()               # display weight histgram
                visualizer.display_current_visuals(model.get_current_visuals(),total_iters,vminmax=None)
                net_dict = model.get_networks()       # display the weight histgram
                visualizer.display_weight_hist(net_dict,total_iters)
                
            total_iters += 1
            epoch_iter += opt.batch_size

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))
        lr=model.update_learning_rate()                     # update learning rates at the end of every epoch.
        visualizer.plot_learning_rate(lr,epoch)

        if epoch % opt.val_epoch_freq ==0:       # validation
            val_start_time = time.time()    # timer for data loading per iteration
            model.eval()
            val_criteria = Criteria('val')
            for i, data in enumerate(val_dataset):  # inner loop within one epoch
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.test()   # forward and calculate visuals
                val_criteria.update(model.get_outputs(),model.get_target())
            model.train()
            accuracies = val_criteria.get_avgs() # retrieve accuracy
            cm = val_criteria.get_cms(normalize=False)
            val_criteria.reset()        # after retrieving the average accuracy, clear the accumulator
            visualizer.plot_current_losses(accuracies, epoch)
            visualizer.plot_cm(cm,epoch)
            print('Validatioin Time Taken: %d sec' % (time.time() - val_start_time))
    print("end of Training")
        
        




