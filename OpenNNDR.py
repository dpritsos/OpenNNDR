# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cosine as cosd

# Open-Set Nearest Neighbor Distance Ration for Multi-Class Classification Framework.


class OpenNNDR(object):

    def __init__(self, slt_ptg, ukwn_slt_ptg, rt_stp, lmda):

        # Definind the hyper-paramter arguments which will be used for the optimisation process...
        # ...of finding the empiricaly optimal rt (ration-therhold) value.
        self.slt_ptg = slt_ptg
        self.ukwn_slt_ptg = ukwn_slt_ptg

        if rt_stp < 0.0 or rt_stp > 1.0:
            raise Exception("The ratio-therhold optimisation step value should in range 0.0 to 1.0")
        self.rt_stp = rt_stp

        if rt_stp < 0.0 or rt_stp > 1.0:
            raise Exception("The ratio-therhold optimisation step value should in range 0.0 to 1.0")

        self.lmda = lmda

    def split(self, y):

        # Calculating the sub-split class sizes for Training, for Known-Testing, Unknown-Testing.
        unq_cls_tgs = np.unique(y)
        ukwn_cls_num = int(np.ceil(unq_cls_tgs.size * self.ukwn_slt_ptg))
        # tr_cls_num = int(np.ceil(unq_cls_tgs.size * self.slt_ptg))
        # kvld_cls_num = int(np.ceil((unq_cls_tgs.size - tr_cls_num) * self.ukwn_slt_ptg))

        # Calculating the number of Unique iteration depeding on Uknown number of splits.
        fc = np.math.factorial
        unq_itr = fc(unq_cls_tgs.size) / (fc(ukwn_cls_num) * fc(unq_cls_tgs.size - ukwn_cls_num))

        # List of arrays of indeces, one list for each unique iteration.
        trn_inds, kvld_inds, ukwn_inds, tgs_combs = list(), list(), list(), list()

        # Starting Random Selection of tags Class Spliting.
        itr = 0
        while True:

            # Selecting the class tags for training.
            uknw_cls_tgs = np.random.choice(unq_cls_tgs, ukwn_cls_num, replace=False)

            # Keeping the combination for verifiing that they are unique.
            uknw_tgs_combs.append(uknw_cls_tgs)

            # Increasing the number of interation only if are all unique, else skip the rest...
            # ... of the this loop and find and other combination in order to be unique.
            ucomb_found = False
            sz = np.vstack(uknw_tgs_combs).shape[0]
            for i in range(sz):
                for j in range(sz):
                    if i != j:
                        if np.array_equal(uknw_tgs_combs[i], uknw_tgs_combs[j]):
                            ucomb_found = True
            if ucomb_found:
                continue
            else:
                itr += 1

            # Getting the Uknown validation-samples indeces of Uknwon class tags.
            ukwn_inds.append(np.where(np.in1d(y, ukwn_cls_tags) == True)[0])

            # Getting the Uknown class tags.
            known_cls_tgs = unq_cls_tgs[np.where(np.in1d(unq_cls_tgs, uknw_cls_tgs) == False)[0]]

            # Spliting the indeces of Known class tags to Training and Validation.
            knwn_idns = np.where(np.in1d(y, known_cls_tgs) == True)[0]
            knwn_idns_num = knwn_idns_num.shape[0]
            tr_idns_num = int(np.ceil(knwn_idns_num * self.slt_ptg))

            # Suffling the indeces before Spliting to Known Training/Validation splits.
            np.random.shuffle(knwn_idns)

            # Getting the training-samples indeces.
            trn_inds.append(knwn_idns[0:knwn_idns_num])

            # Getting the known validation-samples indeces.
            kvld_inds.append(knwn_idns[knwn_idns_num::])

            # When unique iteration have reached the requiered number.
            if itr == unq_itr:
                break

        return trn_inds, kvld_inds, ukwn_inds, unq_cls_tgs

    def score_rt(self, kvld_pre, uknw_pre, kvld_exp, uknw_exp):

        # Normilized Accuracy will be used for this implementation. That is, for the multi-class...
        # ...classification, the correct-prediction over the total known and unkown predictions...
        # ...respectively. The normalized-accuracy NA is the weightied sum of the two accuracies.
        crrt_knw = np.sum(np.equal(kvld_pre, kvld_exp))
        uknw_crrt = np.sum(np.equal(uknw_pre, uknw_exp))

        # Calculating Known-Samples Accuracy and Uknown-Samples Accuracy.
        AKS = crrt_knw / float(kvld_pre.size)
        AUS = uknw_crrt / float(uknw_pre.size)

        # Calculating (and returing) the Nromalized Accuracy.
        return self.lmda * AKS + (1 - self.lmda) * AUS

    def fit(self, X, y):

        # Spliting the Training and Validation (Known and Uknown).
        trn_inds_lst, kvld_inds_lst, ukwn_inds_lst, unq_cls_arr = self.split(y)

        # Optimising the rt threshold for every split. Keeping the rt with the best NA.

        rtz = np.zeros(trn_inds_lst, dtype=np.float)
        NAz = np.zeros(trn_inds_lst, dtype=np.float)

        for rt_i, rt in enumerate(np.arange(0.5, 1.0, self.rt_stp)):

            # Calculating prediction per split.

            kvld_pre, uknw_pre, kvld_exp, uknw_exp = list(), list()

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

                # Classifing validation samples (Known and Uknown).
                kvld_Ds = np.zeros((len(ukwn_inds_lst), kvld_inds.size), dtype=np.float)
                pre_kvld = np.zeros((len(ukwn_inds_lst), kvld_inds.size), dtype=np.float)
                ukwn_Ds = np.zeros((len(ukwn_inds_lst), ukwn_inds.size), dtype=np.float)
                pre_ukwn = np.zeros((len(ukwn_inds_lst), ukwn_inds.size), dtype=np.float)

                for i, ctg in enumerate(unq_cls_tgs):

                    # For this class-tag training inds.
                    cls_tr_inds = np.where(y[trn_inds] == ctg)

                    # Calculating the minimum distancies.
                    kvld_Ds[i, :] = 1.0 - np.matmul(
                        norm_X[cls_tr_inds, :], norm_X[kvld_inds, :].T
                    )

                    ukwn_Ds[i, :] = 1.0 - np.matmul(
                        norm_X[cls_tr_inds, :], norm_X[ukwn_inds, :].T
                    )

                # ###Calculating R and classify based on this rt

                # Getting the first min distance.
                min_kvld_idx = np.argmin(kvld_Ds, axis=1)  # == min_kvld_idx
                min_ukwn_idx = np.argmin(ukwn_Ds, axis=1)  # == min_ukwn_idx
                kvld_mins = kvld_Ds[min_kvld_idx]
                min_ukwn = ukwn_Ds[min_ukwn_idx]

                # Setting Inf the fist min distances posistion for finding the second mins.
                kvld_mins[min_kvld_idx] = np.Inf
                ukwn_mins[min_ukwn_idx] = np.Inf

                # Calculating R rationz.
                knR = min_kvld / np.min(kvld_mins, axis=1)
                uknR = min_ukwn / np.min(ukwn_mins, axis=1)

                # Calculating the Predicition based on this rt threshold.
                pre_kvld = min_kvld_idx
                pre_ukwn = min_ukwn_idx
                pre_kvld[np.where(knR > rt)] = 0
                pre_ukwn[np.where(uknR > rt)] = 0

                # Keeping prediction prediction per split and expected per split.
                kvld_pre.append(pre_kvld)
                uknw_pre.append(pre_ukwn)
                kvld_exp.append(y[kvld_inds])
                uknw_exp.append(y[ukwn_inds])

            # Calculating and keeping the NA Score for this rt threshold.
            rtz[rtz_i] = rt
            NAz[rtz_i] = self.score_rt(
                np.hstack(kvld_pre),
                np.hstack(uknw_pre),
                np.hstack(kvld_exp),
                np.hstack(uknw_exp)
            )

        # Separating the samples for each class. The cls_lst is a list of numpy arrays. Every,...
        # ...array is a list of vector for a specifc class tag.
        # cls_lst = [X[np.where(y == ctg)[0], :] for ctg in unq_cls_tgs]

        # Returin the document vector per class and the rt that maximizes NA.
        return

    def predict(self, X):
        pass

if __name__ == '__main__':

    onndr = OpenNNDR(slt_ptg=0.5, ukwn_slt_ptg=0.3, rt_stp=0.1)

    print onndr.split(np.random.randint(10, size=100))[1]
