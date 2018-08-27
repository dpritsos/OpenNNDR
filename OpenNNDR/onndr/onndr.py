
import numpy as np
import time as tm
from ..dsmeasures.dsmeasures import eudis_2d, cosdis_2d


# Open-Set Nearest Neighbor Distance Ration for Multi-Class Classification Framework.
class OpenNNDR(object):

    def __init__(self, slt_ptg, ukwn_slt_ptg, rt_lims_stp, lmda):

        # Initilising the rt.
        self.rt = 0.0

        # Definind the hyper-paramter arguments which will be used for the optimisation process...
        # ...of finding the empiricaly optimal rt (ration-therhold) value.
        self.slt_ptg = slt_ptg
        self.ukwn_slt_ptg = ukwn_slt_ptg

        if (rt_lims_stp[0] <= 0.0 or rt_lims_stp[0] >= 1.0) and \
            (rt_lims_stp[1] <= 0.0 or rt_lims_stp[1] >= 1.0) and \
                (rt_lims_stp[2] < 0.0 or rt_lims_stp[2] > 1.0):
            raise Exception(
                "The ratio-therhold optimisation step and limits value" +
                "should in range 0.0 to 1.0"
            )
        self.rt_lims_stp = np.array(rt_lims_stp)

        if lmda < 0.0 or lmda > 1.0:
            raise Exception("The lamda valid range is 0.0 to 1.0")
        self.lmda = lmda

    def fit(self, X, y):

        # NOTE: Hopfully this will always prevent potential "Overflow errors" while...
        # calculations from numpy and from cython C level functions.
        X = np.array(X, dtype=np.float64)

        # Spliting the Training and Validation (Known and Uknown).
        trn_inds_lst, kvld_inds_lst, ukwn_inds_lst, unq_ctg_arr = self.split(y)

        # Calculating all the distance between Training and Validation samples (and Uknown samples)
        kvld_minds_pcls_pslt, ukwn_minds_pcls_pslt = list(), list()

        for trn_inds, kvld_inds, ukwn_inds in zip(trn_inds_lst, kvld_inds_lst, ukwn_inds_lst):

            # Getting the unique training-set class tags for this split.
            unq_trn_ctgs = np.unique(y[trn_inds])

            # Classifing validation samples (Known and Uknown).
            kvld_mds_pcls = np.zeros((unq_trn_ctgs.size, kvld_inds.size), dtype=np.float64)
            pre_kvld = np.zeros((unq_trn_ctgs.size, kvld_inds.size), dtype=np.float64)
            ukwn_mds_pcls = np.zeros((unq_trn_ctgs.size, ukwn_inds.size), dtype=np.float64)
            pre_ukwn = np.zeros((unq_trn_ctgs.size, ukwn_inds.size), dtype=np.float64)

            # start_tm = tm.time()

            # print unq_trn_ctgs
            for i, ctg in enumerate(unq_trn_ctgs):

                # For this class-tag training inds.
                cls_tr_inds = np.where(y[trn_inds] == ctg)[0]

                # Calculating the distancies.
                cdists = np.array(cosdis_2d(X[cls_tr_inds, :], X[kvld_inds, :]), dtype=np.float64)
                kvld_mds_pcls[i, :] = np.min(cdists, axis=0)

                cdists = np.array(cosdis_2d(X[cls_tr_inds, :], X[ukwn_inds, :]), dtype=np.float64)
                ukwn_mds_pcls[i, :] = np.min(cdists, axis=0)

            # timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
            # print "Time elapsed : %d:%d:%d:%d" % timel

            # Keeping distances, per class, per split.
            kvld_minds_pcls_pslt.append(kvld_mds_pcls)
            ukwn_minds_pcls_pslt.append(ukwn_mds_pcls)

        # Calculating minimum distances and R rations per Split.
        min_kvld_idx_l, min_ukwn_idx_l, kvld_min_l, min_ukwn_l = [], [], [], []
        knR_l, uknR_l = [], []

        for kvld_mds_pcls, ukwn_mds_pcls in zip(kvld_minds_pcls_pslt, ukwn_minds_pcls_pslt):

            # print kvld_mds_pcls
            # Getting the first min distance.
            min_kvld_idx = np.argmin(kvld_mds_pcls, axis=0)  # == min_kvld_idx
            min_ukwn_idx = np.argmin(ukwn_mds_pcls, axis=0)  # == min_ukwn_idx
            kvld_min = kvld_mds_pcls[min_kvld_idx, np.arange(kvld_mds_pcls.shape[1])]
            min_ukwn = ukwn_mds_pcls[min_ukwn_idx, np.arange(ukwn_mds_pcls.shape[1])]

            # Setting Inf the fist min distances posistion for finding the second mins.
            kvld_mds_pcls[min_kvld_idx, np.arange(kvld_mds_pcls.shape[1])] = np.Inf
            ukwn_mds_pcls[min_ukwn_idx, np.arange(ukwn_mds_pcls.shape[1])] = np.Inf

            # Calculating R rationz.
            # Preventing Division by Zero.
            kvld_min_2 = np.min(kvld_mds_pcls, axis=0)
            kvld_min_2[np.where(kvld_min_2 == 0.0)[0]] = 0.000001

            min_ukwn_2 = np.min(ukwn_mds_pcls, axis=0)
            min_ukwn_2[np.where(min_ukwn_2 == 0.0)[0]] = 0.000001

            knR = kvld_min / kvld_min_2
            uknR = min_ukwn / min_ukwn_2

            # Keeping the minimum distances and Rationsz.
            min_kvld_idx_l.append(min_kvld_idx)
            min_ukwn_idx_l.append(min_ukwn_idx)
            kvld_min_l.append(kvld_min)
            min_ukwn_l.append(min_ukwn)
            knR_l.append(knR)
            uknR_l.append(uknR)

        # Calculating R and classify based on this rt

        # Calculating the range of rt values to be selected for optimisation.
        rt_range = np.arange(self.rt_lims_stp[0], self.rt_lims_stp[1], self.rt_lims_stp[2])

        # Optimising the rt threshold for every split. Keeping the rt with the best NA.
        rtz = np.zeros(rt_range.size, dtype=np.float64)
        NAz = np.zeros(rt_range.size, dtype=np.float64)

        print 'init ', NAz

        for rt_i, rt in enumerate(rt_range):

            kwn_pred_pslt, ukwn_perd_pslt, kwn_exp_pslt, ukwn_exp_pslt = [], [], [], []

            for (
                    kvld_inds,
                    ukwn_inds,
                    min_kvld_idx,
                    min_ukwn_idx,
                    kvld_min,
                    min_ukwn,
                    knR, uknR
                ) in zip(
                            kvld_inds_lst,
                            ukwn_inds_lst,
                            min_kvld_idx_l,
                            min_ukwn_idx_l,
                            kvld_min_l,
                            min_ukwn_l,
                            knR_l,
                            uknR_l
                        ):

                # Calculating the Predicition based on this rt threshold.
                # Check this carefully.
                pre_kvld = np.array([unq_trn_ctgs[min_idx] for min_idx in min_kvld_idx])
                # Check this carefully.
                pre_ukwn = np.array([unq_trn_ctgs[min_idx] for min_idx in min_ukwn_idx])

                pre_kvld[np.where(knR > rt)] = 0
                pre_ukwn[np.where(uknR > rt)] = 0

                # Keeping prediction per split and expected per split.
                kwn_pred_pslt.append(pre_kvld)
                ukwn_perd_pslt.append(pre_ukwn)

                kwn_exp_pslt.append(y[kvld_inds])
                ukwn_exp_pslt.append(np.zeros(ukwn_inds.size))

                # print pre_kvld.shape, y[kvld_inds].shape
                # print pre_ukwn.shape, ukwn_inds.shape

            # Calculating and keeping the NA Score for this rt threshold.
            rtz[rt_i] = rt
            NAz[rt_i] = self.score_rt(
                np.hstack(kwn_pred_pslt),
                np.hstack(ukwn_perd_pslt),
                np.hstack(kwn_exp_pslt),
                np.hstack(ukwn_exp_pslt)
            )

        # Separating and keeping the samples for each class. The cls_d is a dictionary of numpy...
        # ...arrays. Every array is a list of vector for a specifc class tag, which is also a...
        # ...key value for the dictionary.
        self.cls_d = dict([(ctg, X[np.where(y == ctg)[0], :]) for ctg in unq_ctg_arr])

        # Keeping the rt that maximizes NA.
        print NAz
        self.rt = rtz[np.argmax(NAz)]

        return self.cls_d, self.rt

    def predict(self, X):
        return self._predict(X, self.cls_d, self.rt)

    def _predict(self, X, cls_d, rt):

        # NOTE: Hopfully this will always prevent potential "Overflow errors" while...
        # ...calculations from numpy and from cython C level functions.
        X = np.array(X, dtype=np.float64)

        # Getting Class-tags cls_d
        cls_tgs = np.sort(cls_d.keys())

        # Classifing validation samples (Known and Uknown).
        pre_minds_pcls = np.zeros((cls_tgs.size, X.shape[0]), dtype=np.float64)
        pre_y = np.zeros_like(pre_minds_pcls)

        for i, ctg in enumerate(cls_tgs):

            # Getting for this class-tag the ALL the class vectors slected in training phase.
            cls_vects = cls_d[ctg]

            # Calculating the distancies of X where are the random samples while testing...
            # ...or validation phase.
            pred_dists_per_class = cosdis_2d(cls_vects, X)

            # Getting the miminum distance values per samples per class.
            pre_minds_pcls[i, :] = np.min(pred_dists_per_class, axis=0)
            # print pre_minds_pcls[i, :]

        # ###Calculating R and classify based on this rt

        # Getting the first min distance.
        minds_idx = np.argmin(pre_minds_pcls, axis=0)
        min_ds = pre_minds_pcls[minds_idx,  np.arange(pre_minds_pcls.shape[1])]
        # print np.sum(np.where((min_ds == 0.0), 1, 0))

        # Setting Inf the fist min distances posistion for finding the second mins.
        pre_minds_pcls[minds_idx, np.arange(pre_minds_pcls.shape[1])] = np.Inf

        # Calculating R rationz.

        # Preventing Division by Zero.
        min_ds_2 = np.min(pre_minds_pcls, axis=0)
        min_ds_2[np.where(min_ds_2 == 0.0)[0]] = 0.000001

        R = min_ds / min_ds_2

        # Calculating the Predicition based on this rt threshold.
        pre_y = np.array([cls_tgs[min_idx] for min_idx in minds_idx])

        pre_y[np.where(R > rt)] = 0

        return pre_y, R

    def split(self, y):

        # Calculating the sub-split class sizes for Training, for Known-Testing,...
        # ...Unknown-Testing.
        unq_cls_tgs = np.unique(y)
        ukwn_cls_num = int(np.ceil(unq_cls_tgs.size * self.ukwn_slt_ptg))

        # Calculating the number of Unique iteration depeding on Uknown number of splits.
        fc = np.math.factorial
        unq_itr = fc(unq_cls_tgs.size) / (fc(ukwn_cls_num) * fc(unq_cls_tgs.size - ukwn_cls_num))

        # Selecting Randomly the Class-tags Splits.
        itr = 0
        knw_uknw_tgs_combs = list()

        while True:

            # Selecting the unkown class tags.
            uknw_cls_tgs = np.random.choice(unq_cls_tgs, ukwn_cls_num, replace=False)

            # Validating the uniqueness of the randomly selected class-tag as uknown split...
            # ...Then increasing the number of interation only if are all unique, else skip the...
            # ...rest of the this loop and find and other combination in order to be unique.
            ucomb_found = False
            for i in range(len(knw_uknw_tgs_combs)):
                if np.array_equal(uknw_cls_tgs, knw_uknw_tgs_combs[i][1]):
                    ucomb_found = True

            if ucomb_found:
                continue
            else:
                itr += 1

            # Getting the known class tags.
            known_cls_tgs = unq_cls_tgs[
                np.where(np.in1d(unq_cls_tgs, uknw_cls_tgs) == False)[0]
            ]

            # Keeping the combinations for unique Known/Uknown class-tags.
            knw_uknw_tgs_combs.append((known_cls_tgs, uknw_cls_tgs))

            # When unique iteration have reached the requiered number.
            if itr == unq_itr:
                break

        # Selecting the Sample Indices based on the Class Known/Uknown tags spliting.

        # Lists of arrays of indeces, one list for each unique iteration.
        trn_inds, kvld_inds, ukwn_inds, tgs_combs = list(), list(), list(), list()

        for known_cls_tgs, uknw_cls_tgs in knw_uknw_tgs_combs:

            # Getting the Uknown validation-samples indeces of Uknwon class tags.
            ukwn_inds.append(np.where(np.in1d(y, uknw_cls_tgs) == True)[0])

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

        return trn_inds, kvld_inds, ukwn_inds, unq_cls_tgs

    def score_rt(onndr, kvld_pre, uknw_pre, kvld_exp, uknw_exp):

        # Normilized Accuracy will be used for this implementation. That is, for the multi-class...
        # ...classification, the correct-prediction over the total known and unkown predictions...
        # ...respectively. The normalized-accuracy NA is the weightied sum of the two accuracies.
        crrt_knw = np.sum(np.equal(kvld_pre, kvld_exp))
        uknw_crrt = np.sum(np.equal(uknw_pre, uknw_exp))

        # Calculating Known-Samples Accuracy and Uknown-Samples Accuracy.
        AKS = crrt_knw / float(kvld_pre.size)
        AUS = uknw_crrt / float(uknw_pre.size)

        # Calculating (and returing) the Nromalized Accuracy.
        # print AKS, AUS
        return (onndr.lmda * AKS) + ((1.0 - onndr.lmda) * AUS)
