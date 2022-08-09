
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from options import config
import pandas as pd
from sklearn.decomposition import PCA


def lrange(a,b,step=1):
    # a<= <b with step
    return list(range(a,b,step))

def origin2material(groups):
    map_dict=dict()
    for new_idx,group in enumerate(groups):
        for old_idx in group:
            map_dict[old_idx]=new_idx
    return np.vectorize(map_dict.get)

def mat_reduce(cm,mat_map):
    N = cm.shape[0]
    y_true=np.asarray([])
    y_pred=np.asarray([])
    for i in range(N):
        for j in range(N):
            if cm[i,j]>0:
                y_true = np.append(y_true,mat_map(i*np.ones(cm[i,j])))
                y_pred = np.append(y_pred,mat_map(j*np.ones(cm[i,j])))
    new_cm = confusion_matrix(y_true,y_pred)
    acc = sum(y_true==y_pred)/y_true.shape[0]
    return new_cm,acc

def plot_cm(cm,material_list,normalize,title=None,cmap=plt.cm.Blues,colorbar=True,yticklabels=None,valrange=None):
        '''plot confusion matrix (cm)

        Args:
            title: (optional) the title of the matrix, defualt None 
            cmap: (optional) adjusting the coloring map, {default: cm.Blues, difference: plt.cm.bwr}
            colorbar: show colorbar
            yticklabels: specific yticklables that different from material_list
            valrange: restrict the value range in (-v,v) or a list of [vmin,vmax]
        '''
        class_num=len(material_list)
        fig, ax = plt.subplots(figsize=(int(class_num//2+5),int(class_num//2)))
        
        if valrange is not None:
            assert(isinstance(valrange,(float,int,list)))
        if isinstance(valrange,(int,float)):
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap,vmin=-valrange,vmax=valrange)
        elif isinstance(valrange,(list)):
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap,vmin=valrange[0],vmax=valrange[1])
        else:
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        if colorbar:
            ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        if not yticklabels:
            yticklabels=material_list
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            ylim=(cm.shape[0]-0.5,-0.5,),
            # ... and label them with the respective list entries
            xticklabels=material_list, yticklabels=yticklabels,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig,ax

class Visualizer(object):
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'tensorboard' for display
    """

    def __init__(self, opt):
        self.save_dir = opt.save_dir
        self.writer = SummaryWriter(log_dir=opt.save_dir,comment=opt.name)
        self.log_name = os.path.join(opt.save_dir, 'loss_log.txt')
        self.material_list=[]
        albedo_list =[]
        ss_list =[]
        for albedo in range(5):     #[0.2,0.4,0.6,0.8,1]:
            albedo_list.append(lrange(albedo*5,albedo*5+5))
            ss_list.append(lrange(albedo,25,5))
            for ss in range(5):     #[0.002,0.005,0.01,0.02,0.05]:
                self.material_list.append('alb%d-ss%d'%(albedo,ss))
        self.albedo_map = origin2material(albedo_list)
        self.ss_map = origin2material(ss_list)

    def plot_run_acc(self,run_acc,index):
        self.writer.add_scalar(run_acc.name+'_acc',run_acc.avg,index)   
    def plot_scalar(self,name,value,index):
        self.writer.add_scalar(name,value,index)   
    
    def plot_learning_rate(self,lr,epoch):
        self.writer.add_scalar('learning rate',lr,epoch)
        message = 'epoch = %d, learning rate = %.7f' % (epoch, lr)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message  
    
    def display_weight_hist(self,net_dict,index):
        for net_name,net in net_dict.items():
            for i, (name, param) in enumerate(net.named_parameters()):
                if 'bn' not in name:
                    self.writer.add_histogram(net_name+name, param, index)

    def plot_current_losses(self, loss_dict, index):
        for loss_name,loss in loss_dict.items():
            self.writer.add_scalar(loss_name,loss,index)    
    
    def plot_pca(self):
        """ plot pca projection for embedded features
        """
        data = pd.read_csv(os.path.join(self.save_dir,'00002/default/tensors.tsv'),sep='\t').values
        label_alb = pd.read_csv(os.path.join(self.save_dir,'00002/default/metadata.tsv'),sep='\t').values[:,0]
        label_ss = pd.read_csv(os.path.join(self.save_dir,'00003/default/metadata.tsv'),sep='\t').values[:,0]

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data)

        colors = ['navy', 'turquoise', 'darkorange','lime','orangered']

        for y,title in [(label_alb,'albedo'),(label_ss,'subsurface')]:
            print(X_pca.shape)
            plt.figure(figsize=(8, 8))
            for color, i, target_name in zip(colors, [0, 1, 2,3,4], [0,1,2,3,4]):
                plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                            color=color, lw=2, label=target_name)
            plt.savefig(os.path.join(self.save_dir,'PCA_{}'.format(title)))


    def plot_embedding(self,model,layer,label='y'):
        embed = getattr(model, layer)
        N = embed.shape[0]
        embed = embed.reshape([N,-1])
        y = getattr(model, label).cpu().numpy()
        material_label = []
        for i in range(N):
            material_label.append(self.material_list[y[i]])
        albedo = y//5
        ss = y%5
        self.writer.add_embedding(embed,metadata=material_label,global_step=1)
        self.writer.add_embedding(embed,metadata=albedo,global_step=2)
        self.writer.add_embedding(embed,metadata=ss,global_step=3)
        self.plot_pca()
            
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self,losses, epoch, iters, t_comp, t_data,t_display):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f,display: %.3f) ' % (epoch, iters, t_comp, t_data,t_display)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def display_current_visuals(self,visual_dict,index,vminmax=(0,1)):
        #initializing fig
        fig=plt.figure(figsize=(8,5),dpi=80)
        ax=fig.add_subplot(1,1,1)
        ax.axis('off')
        for visual_name,visual_list in visual_dict.items():
            if type(visual_list[1]) == torch.Tensor:
                label = visual_list[1].item()
                title = self.material_list[label]
            else:
                title = visual_list[1]
            visual = visual_list[0].cpu().numpy()
            print('drawing:',visual.shape,title)
            if visual.shape[-1]==3:
                ax.imshow(visual)
                ax.set_title(title)
                tag='%s/channel_%d' %(visual_name,0)
                self.writer.add_figure(tag=tag,figure=fig,global_step=index)
                fig.clear(True)
                ax=fig.add_subplot(1,1,1)
                ax.axis('off')
            else:
                for i in range(visual.shape[0]):
                    if vminmax:
                        fig_img = ax.imshow(visual[i],vmin=vminmax[0],vmax=vminmax[1],cmap='gray')
                    else:
                        fig_img = ax.imshow(visual[i],cmap='gray')
                    ax.set_title(title)
                    fig.colorbar(fig_img)
                    tag='%s/channel_%d' %(visual_name,i)
                    self.writer.add_figure(tag=tag,figure=fig,global_step=index)
                    fig.clear(True)
                    ax=fig.add_subplot(1,1,1)
                    ax.axis('off')
        plt.close()
    
    def plot_cm(self, cm_dict,epoch=0):
        '''
        cm_dict:{name:[cm,normalize]} 
        '''
        for cm_name,cm_list in cm_dict.items():
            cm,normalize = cm_list
            fig,ax = plot_cm(cm,self.material_list,normalize,cm_name)
            self.writer.add_figure(tag='cm/'+cm_name,figure=fig,global_step=epoch)
            print(cm)
            print('subsurface:\n',mat_reduce(cm,self.ss_map))
            print('albedo:\n',mat_reduce(cm,self.albedo_map))
        plt.close()    
        
class VisualizerCapture(Visualizer):
    def __init__(self, opt):
        super().__init__(opt)
        self.material_list=config.material_dict[opt.select_materials]
        self.group_map,self.group_material_list = config.origin2materialCapture(config.material_group,self.material_list)
    
    def plot_cm(self, cm_dict,epoch=0):
        '''
        cm_dict:{name:[cm,normalize]} 
        '''
        for cm_name,cm_list in cm_dict.items():
            cm,normalize = cm_list
            fig,ax = plot_cm(cm,self.material_list,normalize,cm_name)
            self.writer.add_figure(tag='cm/'+cm_name,figure=fig,global_step=epoch)
            print(cm)
            print('material_group:\n',mat_reduce(cm,self.group_map))
            # print('albedo:\n',mat_reduce(cm,self.albedo_map))
        plt.close()    