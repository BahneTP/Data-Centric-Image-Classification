from absl import app
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
import logging
import traceback

class OracleOneShot(AlgorithmSkelton):
    def __init__(self):
        name = "oracle_oneshot"
        AlgorithmSkelton.__init__(self, name)

    def run(self, ds, oracle, dataset_info, v_fold, num_annos, percentage_labeled):
        try:
            k = len(dataset_info.classes) # number classes

            paths, _ = ds.get_training_subsets('all')
            
            unlabaled=0
            test=0
            train=0
            val=0

            for i, path in enumerate(paths):
                org_split = ds.get(path, 'original_split')
                split = ds.get(path, 'split')
                soft_label = ds.get(path, 'soft_gt')

                if split == "unlabeled":
                    unlabaled +=1
                    oracle_label = [float(x) for x in oracle.get_soft_gt(path, 1)]
                    if i == 0:
                        print(oracle_label)
                    ds.update_image(path, org_split, oracle_label)

                if split == "train":
                    continue

                elif split == "val":
                    continue

                elif split == "test":
                    test+=1
                    ds.update_image(path, split, k*[0])

            print(unlabaled,train,val,test)

        except Exception:
            logging.error(traceback.format_exc())

        return ds

def main(argv):
    alg = OracleOneShot()
    alg.apply_algorithm()
    alg.report.show()

if __name__ == '__main__':
    app.run(main)
