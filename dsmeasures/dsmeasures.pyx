# -*- coding: utf-8 -*-
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cyhton: wraparound=False

from cython.parallel import parallel, prange
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "math.h":
    cdef double sqrt(double x) nogil
    cdef double acos(double x) nogil


cpdef double [:, ::1] cosdis_2d(double [:, ::1] m1, double [:, ::1] m2):

    cdef:
        # Matrix index variables.
        Py_ssize_t i, j, k, iz, jz

        # Matrices dimentions intilized variables.
        Py_ssize_t m1_I = m1.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2_I = m2.shape[0]
        Py_ssize_t m2_J = m2.shape[1]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [::1] m1_norms
        double [::1] m2_norms
        double [:, ::1] csdis_vect

        # Definding Pi constant.
        double pi = 3.14159265

    # Creating the temporary cython arrays.
    m1_norms = cvarray(shape=(m1_I,), itemsize=sizeof(double), format="d")
    m2_norms = cvarray(shape=(m2_I,), itemsize=sizeof(double), format="d")
    csdis_vect = cvarray(shape=(m1_I, m2_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for iz in range(m1_I):
            m1_norms[iz] = 0.0

        for iz in range(m2_I):
            m2_norms[iz] = 0.0

        for iz in range(m1_I):
            for jz in range(m2_I):
                csdis_vect[iz, jz] = -1.0

        # Calculating the Norms for the first matrix.
        for i in prange(m1_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m1_J):
                m1_norms[i] += m1[i, j] * m1[i, j]

            # Calculating the Square root of the sum
            m1_norms[i] = sqrt(m1_norms[i])

            # Preventing Division by Zero.
            if m1_norms[i] == 0.0:
                m1_norms[i] = 0.000001


        # Calculating the Norms for the second matrix.
        for i in prange(m2_I, schedule='guided'):

            # Calculating Sum.
            for j in range(m2_J):
                m2_norms[i] += m2[i, j] * m2[i, j]

            # Calculating the Square root of the sum
            m2_norms[i] = sqrt(m2_norms[i])

            # Preventing Division by Zero.
            if m2_norms[i] == 0.0:
                m2_norms[i] = 0.000001

        # Calculating the cosine distances product.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i in prange(m1_I, schedule='guided'):

            for j in range(m2_I):

                # Calculating the elemnt-wise sum of products.
                for k in range(m1_J):
                    csdis_vect[i, j] += m1[i, k] * m2[j, k]

                # Normalizing with the products of the respective vector norms.
                csdis_vect[i, j] = csdis_vect[i, j] / (m1_norms[i] * m2_norms[j])

                # Getting Cosine Distance.
                csdis_vect[i, j] =  acos(csdis_vect[i, j]) / pI

    return csdis_vect


cpdef double [:, ::1] eudis_2d(double [:, ::1] m1, double [:, ::1] m2):

    cdef:
        # Matrix index variables.
        Py_ssize_t i, j, k, iz, jz

        # Matrices dimentions intilized variables.
        Py_ssize_t m1_I = m1.shape[0]
        Py_ssize_t m1_J = m1.shape[1]
        Py_ssize_t m2_I = m2.shape[0]

        # MemoryViews for the cython arrays used for sotring the temporary and...
        # ...to be retured results.
        double [:, ::1] eudis_vect

    # Creating the temporary cython arrays.
    eudis_vect = cvarray(shape=(m1_I, m2_I), itemsize=sizeof(double), format="d")

    # The following operatsion taking place in the non-gil and parallel...
    # ...openmp emviroment.
    with nogil, parallel():

        # Initilising temporary storage arrays. NOTE: This is a mandatory process because as...
        # ...in C garbage values can case floating point overflow, thus, peculiar results...
        # ...like NaN or incorrect calculatons.
        for iz in range(m1_I):
            for jz in range(m2_I):
                eudis_vect[iz, jz] = 0.0

        # Calculating the euclidian distances amogst all vectros of both matrices.
        # NOTE: The m2 matrix is expected to be NON-trasposed but it will treated like it.
        for i in prange(m1_I, schedule='guided'):

            for j in range(m2_I):

                # Calculating the elemnt-wise sum of products.
                for k in range(m1_J):
                    eudis_vect[i, j] += (m1[i, k] - m2[j, k]) * (m1[i, k] - m2[j, k])

                # Normalizing with the products of the respective vector norms.
                eudis_vect[i, j] = sqrt(eudis_vect[i, j])

    return eudis_vect
