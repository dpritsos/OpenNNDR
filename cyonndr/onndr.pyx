# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport abort, malloc, free
from cython.parallel import parallel, prange
from cython.view cimport array as cvarray

import time as tm

cdef extern from "math.h":
    cdef double sqrt(double x) nogil

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

        # Spliting the Training and Validation (Known and Uknown).
        trn_inds_lst, kvld_inds_lst, ukwn_inds_lst, unq_ctg_arr = self.split(y)

        # Calculating the range of rt values to be selected for optimisation.
        rt_range = np.arange(self.rt_lims_stp[0], self.rt_lims_stp[1], self.rt_lims_stp[2])

        # Optimising the rt threshold for every split. Keeping the rt with the best NA.
        rtz = np.zeros(rt_range.size, dtype=np.float)
        NAz = np.zeros(rt_range.size, dtype=np.float)

        for rt_i, rt in enumerate(rt_range):

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

                start_tm = tm.time()

                # Normilize all data for caclulating faster the Cosine Distance/Similarity.
                # trvl_X = X[np.hstack([trn_inds, kvld_inds, ukwn_inds])]
                # trvl_XT = trvl_X.T
                # trvl_XT = trvl_XT.copy(order='C')
                # norm_X = np.divide(
                #         trvl_X,
                #         np.sqrt(
                #             np.diag(self.dot2d(trvl_X, trvl_XT)),
                #             dtype=np.float
                #         ).reshape(trvl_X.shape[0], 1)
                #     )


                for i, ctg in enumerate(unq_trn_ctgs):

                    # For this class-tag training inds.
                    cls_tr_inds = np.where(y[trn_inds] == ctg)[0]

                    # Calculating the distancies.
                    cdists = np.zeros(cls_tr_inds.size, dtype=np.float)
                    cdists = cosdis_2d(X[cls_tr_inds, :], X[kvld_inds, :])
                    print cdists
                    kvld_mds_pcls[i, :] = np.min(cdists, axis=0)

                    cdists = np.zeros(cls_tr_inds.size, dtype=np.float)
                    cdists = cosdis_2d(X[cls_tr_inds, :], X[ukwn_inds, :])
                    ukwn_mds_pcls[i, :] = np.min(cdists, axis=0)

                timel = tm.gmtime(tm.time() - start_tm)[3:6] + ((tm.time() - int(start_tm))*1000,)
                print "Time elapsed : %d:%d:%d:%d" % timel


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
        self.rt = rtz[np.argmax(NAz)]

        return self.cls_d, self.rt

    def predict(self, X, cls_d, rt):

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
            pre_ds_pcls = 1.0 - np.divide(np.matmul(cls_d[ctg], X.T), clsd_X_nf)

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
        pre_y[np.where(R > rt)] = 0

        return pre_y, R

    def split(onndr, y):

        # Calculating the sub-split class sizes for Training, for Known-Testing, Unknown-Testing.
        unq_cls_tgs = np.unique(y)
        ukwn_cls_num = int(np.ceil(unq_cls_tgs.size * onndr.ukwn_slt_ptg))

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
            tr_idns_num = int(np.ceil(knwn_idns_num * onndr.slt_ptg))

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


cdef double [:, ::1] cosdis_2d(double [:, ::1] m1, double [:, ::1] m2):

    # Matrix index variables.
    cdef:
        unsigned int i, j, k
        unsigned int m1_I = m1.shape[0]
        unsigned int m1_J = m1.shape[1]
        unsigned int m2_I = m1.shape[0]
        unsigned int m2_J = m2.shape[1]
        double [::1] m1_norms
        double [::1] m2_norms
        double [:, ::1] csdis_vect
        # double csdis_vect[m1_I][m2_I]
        # double m1_norms[m1_I]
        # double m2_norms[m1_J]

    m1_norms = cvarray(shape=(m1_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(m1_I, m2_J), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gill and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Calculating the Norms for the first matrix.
        # m1_norms = <double *> malloc(sizeof(double) * m1_I)
        # if m1_norms == NULL:
        #     abort()

        for i in prange(m1_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m1_J):
                m1_norms[i] += m1[i, j] * m1[i, j]

            # Calculating the Square root of the sum
            m1_norms[i] = sqrt(m1_norms[i])

        # Calculating the Norms for the second Trasposed matrix.
        # NOTE: It is expected to be trasposed.
        # m2_norms = <double *> malloc(sizeof(double) * m2_J)
        # if m2_norms == NULL:
        #     abort()

        for j in prange(m2_J, schedule='guided'):

            # Calculating Sum.
            for i in range(m2_I):
                m2_norms[j] += m1[j, i] * m1[j, i]

            # Calculating the Square root of the sum
            m2_norms[j] = sqrt(m1_norms[j])

        # Calculating the cosine distances product.
        for i in prange(m1_I, schedule='guided'):
            for j in range(m2_J):

                # Calculating the elemnt-wise sum of products.
                for k in range(m1_J):
                    csdis_vect[i, j] += m1[i, k] * m2[k, j]

                # Normalizing with the products of the respective vector norms.
                csdis_vect[i, j] = csdis_vect[i, j] / (m1_norms[i] * m2_norms[j])

        # Giving the temporary stack memory back to the OS.
        # free(m1_norms)
        # free(m2_norms)
    return csdis_vect
