import numpy as np
from sklearn.metrics import confusion_matrix

class  Criteria(object):
    def __init__(self,name=''):
        self.meters_dict = {}
        self.name = name

    def update(self,output_dict,target):
        '''
        params:
            output_dict: {name:tensor} pairs
            target: corresponding target tensor
        '''
        target = target.cpu()
        for name,output in output_dict.items():
            output = output.cpu()
            _,pred = output.max(dim=1)
            name_ins = self.name + name     ## instance name

            if name_ins not in self.meters_dict.keys():      #create average meter if not registered before
                self.meters_dict[name_ins]=AverageMeter()

            self.meters_dict[name_ins].update(pred,target)
            
    def reset(self):
        for name,avg_meter in self.meters_dict.items():
            avg_meter.reset()

    def get_avgs(self):
        avg_dict={}
        for name,avg_meter in self.meters_dict.items():  # retrive the corresponding values 
            avg_dict[name]=avg_meter.get_avg()
        return avg_dict

    def get_cms(self,normalize):
        cm_dict={}
        for name,avg_meter in self.meters_dict.items():  # retrive the corresponding values 
            cm_dict[name]=[avg_meter.get_cm(normalize),normalize]
        return cm_dict
        
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        '''
        key members:
            y_true: true labels in the form of numpy N
            y_pred: prediced labels in the form of numpy N
        '''
        self.reset()

    def reset(self):
        self.y_pred = np.array([])
        self.y_true = np.array([])

    def update(self, pred,target, n=1):
        """update the output list and target list

        args:
            pred: N torch cpu, same as y_pred
            target: N   torch cpu, same as y_true
        """
        
        y_pred = pred.numpy()
        y_true = target.numpy()

        self.y_pred = np.append(self.y_pred,y_pred)
        self.y_true = np.append(self.y_true,y_true)

    def get_avg(self):
        # print(self.y_pred.shape,self.y_true.shape,self.data_mask.shape)
        sum = (self.y_pred==self.y_true).sum()
        avg = sum / self.y_pred.shape[0]
        return avg

    def get_cm(self,normalize=False):
        """ calculate confusion matrix (cm) from true/predicted labels

        Returns:
            cm: A len(classes)*len(classes) confusion matrix  
        """
        # Compute confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # classes = classes[unique_labels(y_true, y_pred)]   # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        return cm

