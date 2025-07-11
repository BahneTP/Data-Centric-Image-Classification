from absl import app
from src.datasets.common.dataset_skeleton import DatasetSkeleton
import logging
import traceback
import numpy as np
from src.util.json import DatasetDCICJson
from absl import flags
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton


FLAGS = flags.FLAGS

# here you could define some method specific method parameters
# e.g flags.DEFINE_boolean(name='soft', help='Use soft labels for training.', default=False)

class RandomMethod(AlgorithmSkelton):

    def __init__(self):
        # define a name here which will be used to store all result
        # this name might not include the following symbol: -
        name = 'random'
        AlgorithmSkelton.__init__(self,name)

    def run(self, ds , oracle, dataset_info, v_fold,num_annos,percentage_labeled):
        # implementation of run method which should update the given dataset ds
        # the parameters for creating the dataset are given as v_fold (validation fold number),
        # num_annos (number of annotations per image), percentage_labeled (percentage of labeled data)
        # information about the dataset is provided with dataset_info and the orcale
        # can be used to create additional annotations at the cost of increasing the budget

        # save guard against bugs
        try:

            # add your fancy method here
            # we will only get some paths for all datasets
            paths, targets = ds.get_training_subsets('all')

            # instead of'all' you could also use 'train' or 'val' to create these specific subsets
            # paths are the image paths and targets the expected GT data

            # dont forget to seed your code, so that it may reproduce
            np.random.seed(0)


            # create random new labels
            k = len(dataset_info.classes) # number classes
            n = len(paths) # number samples
            random_labels = np.random.rand(n, k)
            temperature = 0.1
            random_labels = np.exp(random_labels / temperature) / np.sum(np.exp(random_labels / temperature), axis=1, keepdims=True)

            # reiterate over all elements and reassign them the desired value
            for i, path in enumerate(paths):
                org_split = ds.get(path, 'original_split')  # determine original split before they were called unlabedl in the intitilization steps
                split = ds.get(path, 'split')
                # label after initilization
                # this value only is valid for split 'train' or 'val'
                soft_label = ds.get(path,'soft_gt')
                if split == "unlabeled":
                  # get random label
                  # update split to original so that it may be used for training the evaluation model
                  # unlabeled data will be ignored by the evaluation model
                  labs =  [float(temp) for temp in random_labels[i]]
                  ds.update_image(path, org_split, labs)
                elif split == 'test':
                  # get random label
                  # test data is ignored by the evaluation model
                  # this data is used to calculate 'input_kl' and so on
                  # update the values with your method to get a first estimate before training the evaluation model
                  ds.update_image(path, split, [float(temp) for temp in random_labels[i]])
                else:
                  # you can call the oracle to get an additional estimate of the gt value
                  # please be aware that this increases your budget usage
                  # in this example we call the functional for all intitialized values this it doubles the budget from the initialized percentage labeled
                  # in the example below from 0.5 to a total budget of 1
                  oracle_label = oracle.get_soft_gt(path, 1)

                  # example of combining the new oracle label with the initialized value
                  label = list([(sofl+secL)/2 for sofl,secL in zip(soft_label,oracle_label)])
                  # this example would basically be the same as intializing 50% of the labels with 2 annotations


                  # we ignore the label and just set a random label
                  ds.update_image(path, split, [float(temp) for temp in random_labels[i]])

        except Exception as e:
            logging.error(traceback.format_exc())

        return ds

def main(argv):
    """
       Apply only initial annotation
       :return:
    """


    alg = RandomMethod()
    alg.apply_algorithm()

    alg.report.show()

if __name__ == '__main__':
    app.run(main)