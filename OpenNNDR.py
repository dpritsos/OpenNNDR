# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cosine as cosd

# Open-Set Nearest Neighbor Distance Ration for Multi-Class Classification Framework.


class OpenNNDR(object):

    def __init__(self, slt_ptg, ukwn_slt_ptg, rt_stp, lmda):

        # Initilising the rt and dictionary of the class vectors arrays.
        self.rt = 0.0
        self.cls_d = dict()

        # Definind the hyper-paramter arguments which will be used for the optimisation process...
        # ...of finding the empiricaly optimal rt (ration-therhold) value.
        self.slt_ptg = slt_ptg
        self.ukwn_slt_ptg = ukwn_slt_ptg

        if rt_stp < 0.0 or rt_stp > 1.0:
            raise Exception("The ratio-therhold optimisation step value should in range 0.0 to 1.0")
        self.rt_stp = rt_stp

        if lmda < 0.0 or lmda > 1.0:
            raise Exception("The lamda valid range is 0.0 to 1.0")
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

        # Init Selecting the class tags for training.
        uknw_cls_tgs = np.random.choice(unq_cls_tgs, ukwn_cls_num, replace=False)
        itr = 0
        uknw_tgs_combs = list()

        while True:

            # Selecting the class tags for training.
            uknw_cls_tgs = np.random.choice(unq_cls_tgs, ukwn_cls_num, replace=False)

            # Increasing the number of interation only if are all unique, else skip the rest...
            # ... of the this loop and find and other combination in order to be unique.
            ucomb_found = False
            sz = len(uknw_tgs_combs)
            for i in range(sz):
                if np.array_equal(uknw_cls_tgs, uknw_tgs_combs[i]):
                    ucomb_found = True

            # Keeping the combination for verifiing that they are unique.
            uknw_tgs_combs.append(uknw_cls_tgs)

            if ucomb_found:
                continue
            else:
                itr += 1

            # Getting the Uknown validation-samples indeces of Uknwon class tags.
            ukwn_inds.append(np.where(np.in1d(y, uknw_cls_tgs) == True)[0])

            # Getting the Uknown class tags.
            known_cls_tgs = unq_cls_tgs[np.where(np.in1d(unq_cls_tgs, uknw_cls_tgs) == False)[0]]

            # Spliting the indeces of Known class tags to Training and Validation.
            knwn_idns = np.where(np.in1d(y, known_cls_tgs) == True)[0]
            knwn_idns_num = knwn_idns.shape[0]
            tr_idns_num = int(np.ceil(knwn_idns_num * self.slt_ptg))

            # Suffling the indeces before Spliting to Known Training/Validation splits.
            np.random.shuffle(knwn_idns)

            # Getting the training-samples indeces.
            trn_inds.append(knwn_idns[0:tr_idns_num])

            # Getting the known validation-samples indeces.
            kvld_inds.append(knwn_idns[tr_idns_num::])

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
        print AKS, AUS
        # Calculating (and returing) the Nromalized Accuracy.
        return (self.lmda * AKS) + ((1.0 - self.lmda) * AUS)

    def fit(self, X, y):

        # Spliting the Training and Validation (Known and Uknown).
        trn_inds_lst, kvld_inds_lst, ukwn_inds_lst, unq_ctg_arr = self.split(y)

        # Calculating the range of rt values to be selected for optimisation.
        rt_range = np.arange(0.5, 1.0, self.rt_stp)

        # Optimising the rt threshold for every split. Keeping the rt with the best NA.
        rtz = np.zeros(rt_range.size, dtype=np.float)
        NAz = np.zeros(rt_range.size, dtype=np.float)

        for rt_i, rt in enumerate(np.arange(0.5, 1.0, self.rt_stp)):

            # Calculating prediction per split.

            kvld_pre, uknw_pre, kvld_exp, uknw_exp = list(), list(), list(), list()

            for trn_inds, kvld_inds, ukwn_inds in zip(trn_inds_lst, kvld_inds_lst, ukwn_inds_lst):

                # Getting the unique training-set class tags for this split.
                unq_trn_ctgs = np.unique(y[trn_inds])

                # Classifing validation samples (Known and Uknown).
                kvld_mds_pcls = np.zeros((unq_trn_ctgs.size, kvld_inds.size), dtype=np.float)
                pre_kvld = np.zeros((unq_trn_ctgs.size, kvld_inds.size), dtype=np.float)
                ukwn_mds_pcls = np.zeros((unq_trn_ctgs.size, ukwn_inds.size), dtype=np.float)
                pre_ukwn = np.zeros((unq_trn_ctgs.size, ukwn_inds.size), dtype=np.float)

                # Normilize all data for caclulating faster the Cosine Distance/Similarity.
                trvl_X = X[np.hstack([trn_inds, kvld_inds, ukwn_inds])]
                norm_X = np.divide(
                        trvl_X,
                        np.sqrt(
                            np.diag(np.dot(trvl_X, trvl_X.T)),
                            dtype=np.float
                        ).reshape(trvl_X.shape[0], 1)
                    )

                for i, ctg in enumerate(unq_trn_ctgs):

                    # For this class-tag training inds.
                    cls_tr_inds = np.where(y[trn_inds] == ctg)[0]

                    # Calculating the distancies.
                    kvld_mds_pcls[i, :] = 1.0 - np.min(
                        np.matmul(norm_X[cls_tr_inds, :], norm_X[kvld_inds, :].T),
                        axis=0
                    )

                    ukwn_mds_pcls[i, :] = 1.0 - np.min(
                        np.matmul(norm_X[cls_tr_inds, :], norm_X[ukwn_inds, :].T),
                        axis=0
                    )

                # ###Calculating R and classify based on this rt

                # Getting the first min distance.
                min_kvld_idx = np.argmin(kvld_mds_pcls, axis=0)  # == min_kvld_idx
                min_ukwn_idx = np.argmin(ukwn_mds_pcls, axis=0)  # == min_ukwn_idx
                kvld_min = kvld_mds_pcls[min_kvld_idx, np.arange(kvld_mds_pcls.shape[1])]
                min_ukwn = ukwn_mds_pcls[min_ukwn_idx, np.arange(ukwn_mds_pcls.shape[1])]

                # Setting Inf the fist min distances posistion for finding the second mins.
                kvld_mds_pcls[min_kvld_idx, np.arange(kvld_mds_pcls.shape[1])] = np.Inf
                ukwn_mds_pcls[min_ukwn_idx, np.arange(ukwn_mds_pcls.shape[1])] = np.Inf

                # Calculating R rationz.
                knR = kvld_min / np.min(kvld_mds_pcls, axis=0)
                uknR = min_ukwn / np.min(ukwn_mds_pcls, axis=0)

                # Calculating the Predicition based on this rt threshold.
                # Check this carefully.
                pre_kvld = np.array([unq_trn_ctgs[min_idx] for min_idx in min_kvld_idx])
                # Check this carefully.
                pre_ukwn = np.array([unq_trn_ctgs[min_idx] for min_idx in min_ukwn_idx])
                pre_kvld[np.where(knR > rt)] = 0
                pre_ukwn[np.where(uknR > rt)] = 0

                # Keeping prediction prediction per split and expected per split.
                kvld_pre.append(pre_kvld)
                uknw_pre.append(pre_ukwn)

                ##############################
                kvld_exp.append(y[kvld_inds])
                uknw_exp.append(np.zeros(ukwn_inds.size))

            # Calculating and keeping the NA Score for this rt threshold.
            rtz[rt_i] = rt
            NAz[rt_i] = self.score_rt(
                np.hstack(kvld_pre),
                np.hstack(uknw_pre),
                np.hstack(kvld_exp),
                np.hstack(uknw_exp)
            )

        # Separating and keeping the samples for each class. The cls_d is a dictionary of numpy...
        # ...arrays. Every array is a list of vector for a specifc class tag, which is also a...
        # ...key value for the dictionary.
        self.cls_d = dict([(ctg, X[np.where(y == ctg)[0], :]) for ctg in unq_ctg_arr])

        # Keeping the rt that maximizes NA.
        print rtz
        self.rt = rtz[np.argmax(NAz)]
        print NAz
        print self.rt

        return self.cls_d, self.rt

    def predict(self, X):
        return self._predict(X, self.cls_d, self.rt)

    def _predict(self, X, cls_d, rt):

        # Getting Class-tags cls_d
        cls_tgs = np.sort(cls_d.keys())

        # Classifing validation samples (Known and Uknown).
        pre_minds_pcls = np.zeros((cls_tgs.size, X.shape[0]), dtype=np.float)
        pre_y = np.zeros_like(pre_minds_pcls)

        # Calculating data normilization factor.
        X_nf = np.sqrt(np.diag(np.matmul(X, X.T)), dtype=np.float)

        for i, ctg in enumerate(cls_tgs):

            # Calculating Cosine Distance normilization factor.
            cls_d_nf = np.sqrt(np.diag(np.matmul(cls_d[ctg], cls_d[ctg].T)), dtype=np.float)
            clsd_X_nf = np.matmul(
                cls_d_nf.reshape(cls_d_nf.size, 1),
                X_nf.reshape(1, X_nf.size)
            )

            # Calculating the Cosine distancies.
            pre_ds_pcls = np.multiply(1.0 - np.matmul(cls_d[ctg], X.T), clsd_X_nf)

            # Getting the miminum distance values per samples per class.
            pre_minds_pcls[i, :] = np.min(pre_ds_pcls, axis=0)

        # ###Calculating R and classify based on this rt

        # Getting the first min distance.
        minds_idx = np.argmin(pre_minds_pcls, axis=0)
        min_ds = pre_minds_pcls[minds_idx,  np.arange(pre_minds_pcls.shape[1])]

        # Setting Inf the fist min distances posistion for finding the second mins.
        pre_minds_pcls[minds_idx, np.arange(pre_minds_pcls.shape[1])] = np.Inf

        # Calculating R rationz.
        R = min_ds / np.min(pre_minds_pcls, axis=0)

        # Calculating the Predicition based on this rt threshold.
        pre_y = np.array([cls_tgs[min_idx] for min_idx in minds_idx])
        print pre_y
        print cls_tgs
        pre_y[np.where(R > rt)] = 0

        return pre_y

if __name__ == '__main__':

    X, y = list(), list()
    X.append(np.random.multivariate_normal([0.1, 0.1], [[0.007, 0.002], [0.007, 0.005]], 110))
    y.append(np.array([1]*110))
    X.append(np.random.multivariate_normal([5.2, 5.2], [[0.008, 0.001], [0.005, 0.007]], 150))
    y.append(np.array([2]*150))
    X.append(np.random.multivariate_normal([20.5, 20.3], [[0.003, 0.004], [0.009, 0.005]], 300))
    y.append(np.array([3]*300))
    X.append(np.random.multivariate_normal([60.8, 50.1], [[0.005, 0.001], [0.007, 0.005]], 200))
    y.append(np.array([4]*200))
    X.append(np.random.multivariate_normal([1.1, 100.7], [[0.007, 0.007], [0.008, 0.005]], 280))
    y.append(np.array([5]*280))
    X.append(np.random.multivariate_normal([50.9, 7.1], [[0.009, 0.001], [0.006, 0.001]], 230))
    y.append(np.array([6]*230))
    X.append(np.random.multivariate_normal([35.3, 9.6], [[0.001, 0.007], [0.009, 0.007]], 300))
    y.append(np.array([7]*300))
    X.append(np.random.multivariate_normal([80.7, 90.2], [[0.003, 0.006], [0.001, 0.005]], 230))
    y.append(np.array([lllll]*230))
    X.append(np.random.multivariate_normal([10.0, 1000.2], [[0.003, 0.006], [0.001, 0.005]], 430))
    y.append(np.array([0]*430))

    tr_X = np.vstack(X[0:7])
    tr_y = np.hstack(y[0:7])

    onndr = OpenNNDR(slt_ptg=0.5, ukwn_slt_ptg=0.3, rt_stp=0.05, lmda=0.5)

    onndr.fit(tr_X, tr_y)
    print y[7]
    print onndr.predict(X[7])
    print onndr.score_rt(onndr.predict(X[7]), onndr.predict(X[8]), y[7], y[8])
