"""patch based test, show results like cm
"""
import time

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer,VisualizerCapture
from utils.criteria import Criteria
import numpy as np
import os

if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    test_dataset = create_dataset(opt,opt.test_csv)
    print('test images = %d' % (len(test_dataset)))

    # create a visualizer that display/save images and plots
    if opt.dataset_mode.lower() == "patchblender":
        visualizer = Visualizer(opt)
    elif opt.dataset_mode.lower() == "patchcapture":
        visualizer = VisualizerCapture(opt)   
    else:
        raise(NotImplementedError('not implimented for dataset: {}'.format(opt.dataset_mode)))

    total_iters = 0                # the total number of training iterations
    val_start_time = time.time()    # timer for data loading per iteration
    model.eval()
    test_criteria = Criteria('test')
    for i, data in enumerate(test_dataset):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()   # forward and calculate visuals
        test_criteria.update(model.get_outputs(),model.get_target())
    accuracies = test_criteria.get_avgs() # retrieve accuracy
    cm = test_criteria.get_cms(normalize=False)

    np.save(os.path.join(opt.save_dir,'cm'),cm['testout'])
    visualizer.print_current_losses(accuracies,opt.load_epoch,0,0,0,0)
    
    visualizer.plot_cm(cm,opt.load_epoch)
    visualizer.plot_current_losses(accuracies, opt.load_epoch)
    visualizer.display_current_visuals(model.get_current_visuals(),opt.load_epoch,vminmax=None)
    visualizer.plot_embedding(model,'embed')
    
    
    test_criteria.reset()        # after retrieving the average accuracy, clear the accumulator
    print('test Time Taken: %d sec' % (time.time() - val_start_time))
    print("end of Testing cm")





