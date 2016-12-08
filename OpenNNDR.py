# -*- coding: utf-8 -*-

import numpy as np

# Open-Set Nearest Neighbor Distance Ration for Multi-Class Classification Framework.


class OpenNNDR(object):

    def __init__(self, slt_ptg, vld_slt_ptg, rt_stp):

        # Defining the ration-threhold paramter.
        self.rt = 0.0

        # Definind the hyper-paramter arguments which will be used for the optimisation process...
        # ...of finding the empiricaly optimal rt (ration-therhold) value.
        self.slt_ptg = slt_ptg
        self.vld_slt_ptg = vld_slt_ptg
        self.rt_stp = rt_stp

    def split(self, y):

        # Calculating the sub-split class sizes for Training, for Known-Testing, Unknown-Testing.
        unq_cls_tgs = np.unique(y)
        tr_cls_num = int(np.ceil(unq_cls_tgs.size * self.slt_ptg))
        kvld_cls_num = int(np.ceil((unq_cls_tgs.size - tr_cls_num) * self.vld_slt_ptg))

        # Calculating the number of Unique iteration depeding on Testing + Uknown number of splits.
        fc = np.math.factorial
        unq_itr = fc(unq_cls_tgs.size) / (fc(tr_cls_num) * fc(unq_cls_tgs.size - tr_cls_num))

        # List of arrays of indeces, one list for each unique iteration.
        trn_inds, kvld_inds, ukwn_inds, tgs_combs = list(), list(), list(), list()

        # Starting Random Selection of tags Class Spliting.
        itr = 0
        while True:

            # Selecting the class tags for training.
            trn_cls_tgs = np.random.choice(unq_cls_tgs, tr_cls_num, replace=False)

            # Keeping the combination for verifiing that they are unique.
            tgs_combs.append(trn_cls_tgs)

            # Increasing the number of interation only if the are all unique, else skip the rest...
            # ... of the this loop and find and other combination in order to be unique.
            sz = np.vstack(tgs_combs).shape[0]
            for i in range(sz):
                for j in range(sz):
                    if i != j:
                        if np.array_equal(tgs_combs[i], tgs_combs[j]):
                            continue
            itr += 1

            # Getting the validation (known and Uknown) class tags.
            vld_cls_tgs = unq_cls_tgs[np.where(np.in1d(unq_cls_tgs, trn_cls_tgs) == False)[0]]

            # Spliting validation class tags to known and Uknown.
            kvld_cls_tags = vld_cls_tgs[0:kvld_cls_num]
            ukwn_cls_tags = vld_cls_tgs[kvld_cls_num::]

            # Getting the training-samples indeces.
            trn_inds.append(np.where(np.in1d(y, trn_cls_tgs) == True)[0])

            # Getting the validation-samples indeces known and uknown.
            kvld_inds.append(np.where(np.in1d(y, kvld_cls_tags) == True)[0])
            ukwn_inds.append(np.where(np.in1d(y, ukwn_cls_tags) == True)[0])

            # When unique iteration have reached the requiered number.
            if itr == unq_itr:
                break

        return trn_inds, kvld_inds, ukwn_inds, unq_cls_tgs

    def optRT(self):
        pass

    def fit(self, X, y):

        # Spliting the Training and Validation (Known and Uknown).
        trn_inds, kvld_inds, ukwn_inds, unq_cls_tgs = self.split(y)




    def predict(self, X):
        pass

if __name__ == '__main__':

    onndr = OpenNNDR(slt_ptg=0.5, vld_slt_ptg=0.3, rt_stp=0.1)

    print onndr.split(np.random.randint(10, size=100))[1]
