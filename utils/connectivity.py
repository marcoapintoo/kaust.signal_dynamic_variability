# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import linalg, fftpack

def var_to_pdc(A):
    #https://gist.github.com/agramfort/9875439
    p, N, N = A.shape
    n_fft = max(int(2 ** np.ceil(np.log2(p))), 512)
    A2 = np.zeros((n_fft, N, N))
    A2[1:p + 1, :, :] = A  # start at 1 !
    fA = fftpack.fft(A2, axis=0)
    freqs = fftpack.fftfreq(n_fft)
    I = np.eye(N)
    for i in range(n_fft):
        fA[i] = linalg.inv(I - fA[i])
    P = np.zeros((n_fft, N, N))
    sigma = np.ones(N)
    for i in range(n_fft):
        B = fA[i]
        B = linalg.inv(B)
        V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
        V = np.diag(V)  # denominator squared
        P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]
    return P

def fmri_var_to_pdc(A):
    #https://gist.github.com/agramfort/9875439
    A = np.nan_to_num(A)
    p, N = 1, A.shape[0]
    n_fft = max(int(2 ** np.ceil(np.log2(p))), 512)
    A2 = np.zeros((n_fft, N, N))
    A2[1:p + 1, :, :] = A  # start at 1 !
    fA = fftpack.fft(A2, axis=0)
    freqs = fftpack.fftfreq(n_fft)
    I = np.eye(N)
    for i in range(n_fft):
        fA[i] = linalg.inv(I - fA[i])
    P = np.zeros((n_fft, N, N))
    sigma = np.ones(N)
    for i in range(n_fft):
        B = fA[i]
        B = linalg.inv(B)
        V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
        V = np.diag(V)  # denominator squared
        P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]
    return P.mean(axis=0)