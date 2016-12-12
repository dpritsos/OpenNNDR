# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cosine as cosd

# Open-Set Nearest Neighbor Distance Ration for Multi-Class Classification Framework.


class OpenNNDR(object):

    def __init__(self, slt_ptg, vld_slt_ptg, rt_stp, lmda):

        # Defining the ration-threhold paramter.
        self.rt = 0.0

        # Definind the hyper-paramter arguments which will be used for the optimisation process...
        # ...of finding the empiricaly optimal rt (ration-therhold) value.
        self.slt_ptg = slt_ptg
        self.vld_slt_ptg = vld_slt_ptg

        if rt_stp < 0.0 or rt_stp > 1.0:
            raise Exception("The ratio-therhold optimisation step value should in range 0.0 to 1.0")
        self.rt_stp = rt_stp

        if rt_stp < 0.0 or rt_stp > 1.0:
            raise Exception("The ratio-therhold optimisation step value should in range 0.0 to 1.0")

        self.lmda = lmda

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

    def score_rt(self, pre_y, exp_y, kvld_inds, ukwn_inds):

        # Normilized Accuracy will be used for this implementation. That is, for the multi-class...
        # ...classification, the correct-prediction over the total known and unkown predictions...
        # ...respectively. The normalized-accuracy NA is the weightied sum of the two accuracies.
        crrt_knw = np.sum(np.where(np.in1d(pre_y[kvld_inds], exp_y[kvld_inds]), 1.0, 0.0))
        uknw_crrt = np.sum(np.where(np.in1d(pre_y[ukwn_inds], exp_y[ukwn_inds]), 1.0, 0.0))

        # Calculating Known-Samples Accuracy and Uknown-Samples Accuracy.
        AKS = crrt_knw / float(pre_y[kvld_inds].shape[0])
        AUS = uknw_crrt / float(pre_y[ukwn_inds].shape[0])

        # Calculating (and returing) the Nromalized Accuracy.
        return self.lmda * AKS + (1 - self.lmda) * AUS

    def fit(self, X, y):

        # Spliting the Training and Validation (Known and Uknown).
        trn_inds_lst, kvld_inds_lst, ukwn_inds_lst, unq_cls_tgs = self.split(y)

        # Separating the samples for each class. The cls_lst is a list of numpy arrays. Every,...
        # ...array is a list of vector for a specifc class tag.
        # cls_lst = [X[np.where(y == ctg)[0], :] for ctg in unq_cls_tgs]

        # Optimising the rt threshold for every split.
        for trn_inds, kvld_inds, ukwn_inds in zip(trn_inds_lst, kvld_inds_lst, ukwn_inds_lst):

            # Normilize all data for caclulating faster the Cosine Distance/Similarity.
            trvl_X = X[np.hstack([trn_inds, kvld_inds, ukwn_inds])]
            norm_X = np.divide(
                    trvl_X,
                    np.sqrt(
                        np.diag(np.dot(trvl_X, trvl_X.T)),
                        dtype=np.float
                    ).reshape(trvl_X.shape[0], 1)
                )

            # Separating the samples for each class. The cls_lst is a list of numpy arrays...
            # ...Every array is a list of vector for a specifc class tag.
            cls_lst = [X[np.where(y[trn_inds] == ctg)[0], :] for ctg in unq_cls_tgs]

            # Keeping the rt with the besr NA.
            for rt in np.arange(0.5, 1.0, self.rt_stp):

                # Classifing validation samples (Known and Uknown).
                kvld_Dz = np.tril(1.0 - np.matmul(norm_X[trn_inds, :], norm_X[kvld_inds, :].T), -1)
                ukwn_Dz = np.tril(1.0 - np.matmul(norm_X[trn_inds, :], norm_X[ukwn_inds, :].T), -1)
                # tril_indices







                self.rt



    def predict(self, X):
        pass

if __name__ == '__main__':

    onndr = OpenNNDR(slt_ptg=0.5, vld_slt_ptg=0.3, rt_stp=0.1)

    print onndr.split(np.random.randint(10, size=100))[1]
