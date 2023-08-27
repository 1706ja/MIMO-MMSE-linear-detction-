import time
import torch
import numpy as np
import multiprocessing
# from multiprocessing import Pool
import matplotlib.pyplot as plt
from math import sqrt
from scipy.linalg import toeplitz
# Parameters Setting
TestDataLen = 30000
TxAntNum = 8   # Number of transmitting antennas
RxAntNum = 16   # Number of receiving antennas tested
DataLen  = 1    # Length of data sequence
MaxIter = 4     # Max iteration number of iterative algorithms
delta = 0.5     # Damping factor

# SNR setting
SNRdBLow = 0   # Minimal value of SNR in dB
SNRdBHigh = 12   # Maximal value of SNR in dB
SNRIterval = 2  # Interval value of SNR sequence
SNRNum = int((SNRdBHigh-SNRdBLow)/SNRIterval)+1   # Number of SNR sequence
SNRdB = np.linspace(SNRdBLow, SNRdBHigh, SNRNum)
TSratio = 0
RSratio = 0
Kron = np.sqrt(TSratio*RSratio)
# Constellation Setting
ModType = 8
if ModType==2:
    Model = '4QAM'
    Cons = torch.tensor([-1., 1.])
    fnorm = 1/np.sqrt(2)
    bitperSym = 1
    normCons = fnorm*Cons
elif ModType==4:
    Model = '16QAM'
    Cons = torch.tensor([-3., -1., 1., 3.])
    bitCons = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    bitperSym = 2
    fnorm = 1/np.sqrt(10)
    normCons = fnorm*Cons
elif ModType==8:
    Model = '64QAM'
    Cons = torch.tensor([-7., -5., -3., -1., 1., 3., 5., 7.])
    bitCons = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    bitperSym = 3
    fnorm = 1/np.sqrt(42)
    normCons = fnorm*Cons
elif ModType==16:
    Model = '256QAM'
    Cons = torch.tensor([-15., -13., -11., -9., -7., -5., -3., -1., 1., 3., 5., 7., 9., 11., 13., 15.])
    bitCons = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    bitperSym = 4
    fnorm = 1/np.sqrt(170)
    normCons = fnorm*Cons

# Variable Initialization
error_MMSE = np.zeros(SNRNum)
# Neumann Series
error_MMSENSA3 = np.zeros(SNRNum)
error_MMSENSA4 = np.zeros(SNRNum)
# successive-overrelaxation
error_MMSESOR2 = np.zeros(SNRNum)
error_MMSESOR3 = np.zeros(SNRNum)
# Gauss-Siedel
error_MMSEGS3 = np.zeros(SNRNum)
error_MMSEGS4 = np.zeros(SNRNum)
# Conjugate Gradient
error_MMSECG3 = np.zeros(SNRNum)
error_MMSECG4 = np.zeros(SNRNum)



# Correlation channel
def KronChannel(TxAntNum, RxAntNum, DataLen, TSratio, RSratio):
    # 1. Generate Rayleigh channel matrix
    Hiid = np.sqrt(1 / 2) * (torch.randn(DataLen, RxAntNum, TxAntNum) + 1j * torch.randn(DataLen, RxAntNum, TxAntNum))

    # 2. Generate the Transmit UpperMatrix
    indexT = torch.tensor(toeplitz(-torch.arange(TxAntNum), torch.arange(TxAntNum))).unsqueeze(dim=-3)
    randT = torch.rand([DataLen, 1, 1])
    phaseT = torch.exp(1j * randT * np.pi / 2 * indexT)
    ampT = TSratio ** torch.abs(indexT)
    Rt = ampT * phaseT
    Ct = torch.linalg.cholesky(Rt, upper=True)

    # 3. Generate the Receive UpperMatrix
    indexR = torch.tensor(toeplitz(-torch.arange(RxAntNum), torch.arange(RxAntNum))).unsqueeze(dim=-3)
    randR = torch.rand([DataLen, 1, 1])
    phaseR = torch.exp(1j * randR * np.pi / 2 * indexR)
    ampR = RSratio ** torch.abs(indexR)
    Rr = ampR * phaseR
    Cr = torch.linalg.cholesky(Rr, upper=True)

    Hkron = Cr.matmul(Hiid.matmul(Ct.conj().transpose(-1, -2)))

    return Hkron


# Generating Data Sent, Recived, Noise, and Transmitting Matrix
def GenerateTestData(SampleNum, TxAntNum, RxAntNum, DataLen, SNRdBLow, SNRdBHigh):
    x_ = torch.zeros([SampleNum, 2*TxAntNum, DataLen])
    y_ = torch.zeros([SampleNum, 2*RxAntNum, DataLen])
    H_ = torch.zeros([SampleNum, 2*RxAntNum, 2*TxAntNum])
    Nv_ = torch.zeros([SampleNum, 1, 1])
    H = KronChannel(TxAntNum, RxAntNum, SampleNum, TSratio, RSratio)
    for itr in range(SampleNum):
        RandSNRdB = np.random.uniform(low=SNRdBLow, high=SNRdBHigh)
        SNR = 10**(RandSNRdB/10)
        # Generate real and image part of data sequence seperately
        TxDataSyms = torch.randint(0, ModType, size=(2*TxAntNum, DataLen))
        # TxData_r, TxData_i = Modulation(TxDataBits, ModType, Cons)
        TxData_r = Cons[TxDataSyms[:TxAntNum, :]]
        TxData_i = Cons[TxDataSyms[TxAntNum:, :]]
        # Transform complex Tx signals to real
        TxData = torch.cat((TxData_r, TxData_i), dim=0)
        x_[itr, :, :] = TxData
        # TxSymbol = np.concatenate((TxPilot, TxData), 1)
        TxSymbol = TxData
        # Generate channel matrix (Rayleigh channel)
        # Hc = np.sqrt(1/2)*(torch.randn(RxAntNum, TxAntNum) + 1j*torch.randn(RxAntNum, TxAntNum))
        Hc = H[itr]
        # Transform complex channle matrix to real
        HMat = torch.cat((torch.cat((torch.real(Hc), -torch.imag(Hc)), 1),
                          torch.cat((torch.imag(Hc), torch.real(Hc)), 1)), 0).float()
        # Normalize the column of real channel matrix
        HMat /= np.sqrt(torch.norm(HMat)**2/(2*TxAntNum))
        # Data send via the channels without AWGN
        RxSymbol_noAWGN = torch.matmul(HMat, TxSymbol)
        # Calculate the norm of channel matrix
        Hnorm = torch.norm(HMat)**2
        # Noise variance & adding AWGN
        Nv = (1*Hnorm)/(2*SNR*(2*RxAntNum)*fnorm**2)
        RxSymbol = RxSymbol_noAWGN + np.sqrt(Nv)*torch.randn(2*RxAntNum, DataLen)
        Hhat = HMat
        y_[itr, :, :] = RxSymbol
        H_[itr, :, :] = Hhat
        Nv_[itr, :, :] = Nv


    return x_, y_, H_, Nv_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def MMSEtest(x, y, H, Nv):
    error_MMSE = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)

    # MIMO detection (MMSE) using perfect CSI
    Sigma = torch.inverse(HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0))
    xhat = torch.matmul(Sigma, HTy)

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSE += len(comp[0])

    return error_MMSE


def MMSE_NSA3test(x, y, H, Nv):
    error_MMSENSA3 = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    # Diagonal matrix generated, inverse calculated
    E = A - torch.diag_embed(D)
    # k=2
    DinvE = Dinv.matmul(E)
    IDE = torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - DinvE
    DE2 = DinvE.matmul(IDE)
    # k=3
    IDE2 = torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - 0.5*DE2 # weighted factor of 0.5
    DE3 = DinvE.matmul(IDE2)
   

    Ainv_apr = (torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - DE3).matmul(Dinv)

    # MIMO detection (MMSE) using perfect CSI
    Sigma = Ainv_apr
    xhat = torch.matmul(Sigma, HTy)

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSENSA3 += len(comp[0])

    return error_MMSENSA3


def MMSE_NSA4test(x, y, H, Nv):
    error_MMSENSA4 = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    # Diagonal matrix generated, inverse calculated
    E = A - torch.diag_embed(D)
    # k=2
    DinvE = Dinv.matmul(E)
    IDE = torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - DinvE
    DE2 = DinvE.matmul(IDE)
    # k=3
    IDE2 = torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - 0.5*DE2 # weighted factor of 0.5
    DE3 = DinvE.matmul(IDE2)
    # k=4
    IDE3 = torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - 0.5*DE3 # weighted factor of 0.5
    DE4 = DinvE.matmul(IDE3)

    Ainv_apr = (torch.unsqueeze(torch.eye(2 * TxAntNum), 0) - DE4).matmul(Dinv)

    # MIMO detection (MMSE) using perfect CSI
    Sigma = Ainv_apr
    xhat = torch.matmul(Sigma, HTy)

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSENSA4 += len(comp[0])

    return error_MMSENSA4

def MMSESOR2test(x,y,H,Nv):
    errorMMSE_SOR2 = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    # inverse of diagonal matrix
    E = A - torch.diag_embed(D)
    Nv1 = Nv[0].numpy()
    Nv2 = Nv1[0][0]
    # this is because Nv is a 3-dimensional vector, and this
    # method can translate it to a number
    a = RxAntNum / (TxAntNum + Nv2 ** 2)
    b = 1 + sqrt(TxAntNum / RxAntNum)
    b = b ** 2
    w = 2 / (1 + sqrt(2 * a * b))
    # computation of weighted factor
    L = torch.tril(E,-1)
    # lower diagonal matrix
    C = 1/w* torch.diag_embed(D) + L
    C1 = torch.inverse(C)
    f = 1/w*C1
    C2 = C1.matmul(L - 1 / w * E)
    # computation of iteration matrix
    f1 = f.matmul(HTy)
    # computation of iteration vector
    x1 = Dinv.matmul(HTy)
    # pre-computation
    # k = 1
    x2 = C2.matmul(x1) + f1
    # k = 2
    x3 = C2.matmul(x2) + f1
    xhat = x3

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    errorMMSE_SOR2 += len(comp[0])
    return errorMMSE_SOR2


def MMSESOR3test(x,y,H,Nv):
    errorMMSE_SOR3 = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    E = A - torch.diag_embed(D)
    Nv1 = Nv[0].numpy()
    Nv2 = Nv1[0][0]
    # this is because Nv is a 3-dimensional vector, and this
    # method can translate it to a number
    a = RxAntNum / (TxAntNum + Nv2 ** 2)
    b = 1 + sqrt(TxAntNum / RxAntNum)
    b = b ** 2
    w = 2 / (1 + sqrt(2 * a * b))
    # computation of weighted factor
    L = torch.tril(E, -1)
    # lower diagonal matrix
    C = 1 / w * torch.diag_embed(D) + L
    C1 = torch.inverse(C)
    f = 1 / w * C1
    C2 = C1.matmul(L - 1 / w * E)
    # computation of iteration matrix
    f1 = f.matmul(HTy)
    # computation of iteration vector
    x1 = Dinv.matmul(HTy)
    # pre-computation
    # k = 1
    x2 = C2.matmul(x1) + f1
    # k = 2
    x3 = C2.matmul(x2) + f1
    # k = 3
    x4 = C2.matmul(x3) + f1
    xhat = x4

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    errorMMSE_SOR3 += len(comp[0])
    return errorMMSE_SOR3

def MMSEGS3test(x,y,H,Nv):
    error_MMSEGS3 = 0
    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    # the inverse of diagonal matrix
    E = A - torch.diag_embed(D)
    L = torch.tril(E,-1)
    U = E - L
    # lower and upper triangular matrix
    C = torch.inverse(torch.diag_embed(D)+L)
    f = C.matmul(HTy)
    # computation of iteration vector
    C1 =C.matmul(U)
    # computation of iteration matrix
    x1 = Dinv.matmul(HTy)
    # pre-computation
    # k = 1
    x2 = -C1.matmul(x1) + f
    # k = 2
    x3 = -C1.matmul(x2) + f
    # k = 3
    x4 = -C1.matmul(x3) + f
    xhat = x4

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSEGS3 += len(comp[0])
    return error_MMSEGS3

def MMSEGS4test(x,y,H,Nv):
    error_MMSEGS4 = 0
    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    # the inverse of diagonal matrix
    E = A - torch.diag_embed(D)
    L = torch.tril(E, -1)
    U = E - L
    # lower and upper triangular matrix
    C = torch.inverse(torch.diag_embed(D) + L)
    f = C.matmul(HTy)
    # computation of iteration vector
    C1 = C.matmul(U)
    # computation of iteration matrix
    x1 = Dinv.matmul(HTy)
    # pre-computation
    # k = 1
    x2 = -C1.matmul(x1) + f
    # k = 2
    x3 = -C1.matmul(x2) + f
    # k = 3
    x4 = -C1.matmul(x3) + f
    # k = 4
    x5 = -C1.matmul(x4) + f
    xhat = x5

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSEGS4 += len(comp[0])
    return error_MMSEGS4


def MMSECG3test(x,y,H,Nv):
    error_MMSECG3 = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    xhat = Dinv.matmul(HTy)
    r = HTy - A.matmul(xhat)
    p = r
    # k = 1
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of so known "alpha"
    xhat = xhat + torch.mul(alpha,p)
    # improvement of xhat
    beta1 = A.matmul(p)
    beta1 = torch.mul(alpha,beta1)
    r -= beta1
    # update of r
    beta2 = r.transpose(-2, -1).matmul(r)
    beta = beta2 / alpha1
    # computation of so known "beta"
    p = r+torch.mul(p[1],beta[1])
    # update of p
    # k = 2
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of so known "alpha"
    xhat = xhat + torch.mul(alpha,p)
    # improvement of xhat
    beta1 = A.matmul(p)
    beta1 = torch.mul(alpha,beta1)
    r -= beta1
    # update of r
    beta2 = r.transpose(-2, -1).matmul(r)
    beta = beta2 / alpha1
    # computation of so known "beta"
    p = r+torch.mul(p[1],beta[1])
    # update of p
    # k = 3
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of so known "alpha"
    xhat = xhat + torch.mul(alpha,p)
    # improvement of xhat
    # no need to do the upcoming computation since xhat is all we need

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSECG3 += len(comp[0])
    return error_MMSECG3


def MMSECG4test(x, y, H, Nv):
    error_MMSECG4 = 0

    HTy = H.transpose(-2, -1).matmul(y)
    HTH = H.transpose(-2, -1).matmul(H)
    A = HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0)
    D = torch.diagonal(A, dim1=-2, dim2=-1)
    Dinv = torch.diag_embed(1 / D)
    E = A - torch.diag_embed(D)
    xhat = Dinv.matmul(HTy)
    r = HTy - A.matmul(xhat)
    p = r
    # k = 1
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of so known "alpha"
    xhat = xhat + torch.mul(alpha, p)
    # improvement of xhat
    beta1 = A.matmul(p)
    beta1 = torch.mul(alpha, beta1)
    r -= beta1
    # update of r
    beta2 = r.transpose(-2, -1).matmul(r)
    beta = beta2 / alpha1
    # computation of so known "beta"
    p = r + torch.mul(p[1], beta[1])
    # update of p
    # k = 2
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of so known "alpha"
    xhat = xhat + torch.mul(alpha, p)
    # improvement of xhat
    beta1 = A.matmul(p)
    beta1 = torch.mul(alpha, beta1)
    r -= beta1
    # update of r
    beta2 = r.transpose(-2, -1).matmul(r)
    beta = beta2 / alpha1
    # computation of so known "beta"
    p = r + torch.mul(p[1], beta[1])
    # update of p
    # k = 3
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of the known "alpha"
    xhat = xhat + torch.mul(alpha, p)
    # improvement of xhat
    beta1 = A.matmul(p)
    beta1 = torch.mul(alpha, beta1)
    r -= beta1
    # update of r
    beta2 = r.transpose(-2, -1).matmul(r)
    beta = beta2 / alpha1
    # computation of the known "beta"
    p = r + torch.mul(p[1], beta[1])
    # update of p
    # k = 4
    alpha1 = r.transpose(-2, -1).matmul(r)
    alpha2 = p.transpose(-2, -1).matmul(A)
    alpha3 = alpha2.matmul(p)
    alpha = alpha1 / alpha3
    # computation of so known "alpha"
    xhat = xhat + torch.mul(alpha, p)
    # improvement of xhat
    # no need to do the upcoming computation since xhat is all we need

    # calculation of BER
    _, indices = torch.min((xhat - Cons) ** 2, dim=-1, keepdim=True)
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_MMSECG4 += len(comp[0])
    return error_MMSECG4


# set of time domains
t1 = 0
t2 = 0
t3 = 0
t4 = 0
t5 = 0
t6 = 0
t7 = 0
t8 = 0
t9 = 0


if __name__ == "__main__":
    print("cpu count:", multiprocessing.cpu_count(), "\n")
    for nEN in range(SNRNum):
        print(SNRdB[nEN])

        for i in range (5):
            # we do this because the process of "GenerateTestData" sometimes uses
            # too much space, thus, we must need this loop to maintain the
            # number of TestDataLen
            x, y, H, Nv = GenerateTestData(TestDataLen, TxAntNum, RxAntNum,
                                           DataLen, SNRdB[nEN], SNRdB[nEN])
            start = time.time()
            error_MMSE[nEN] = MMSEtest(x, y, H, Nv)

            t1 += time.time() - start


            start = time.time()
            error_MMSENSA3[nEN] = MMSE_NSA3test(x, y, H, Nv)
            t2 += time.time() - start

            start = time.time()
            error_MMSENSA4[nEN] = MMSE_NSA4test(x, y, H, Nv)
            t3 += time.time() - start

            start = time.time()
            error_MMSESOR2[nEN] = MMSESOR2test(x, y, H, Nv)
            t4 += time.time() - start

            start = time.time()
            error_MMSESOR3[nEN] = MMSESOR3test(x, y, H, Nv)
            t5 += time.time() - start

            start = time.time()
            error_MMSEGS3[nEN] = MMSEGS3test(x, y, H, Nv)
            t6 += time.time() - start

            start = time.time()
            error_MMSEGS4[nEN] = MMSEGS4test(x, y, H, Nv)
            t7 += time.time() - start

            start = time.time()
            error_MMSECG3[nEN] = MMSECG3test(x, y, H, Nv)
            t8 += time.time() - start

            start = time.time()
            error_MMSECG4[nEN] = MMSECG4test(x, y, H, Nv)
            t9 += time.time() - start


        print("Testing Time (MMSE):", t1)
        print("Testing Time (MMSE-NSA3):", t2)
        print("Testing Time (MMSE-NSA4):", t3)
        print("Testing Time (MMSE-SOR2):", t4)
        print("Testing Time (MMSE-SOR3):", t5)
        print("Testing Time (MMSE-GS3):", t6)
        print("Testing Time (MMSE-GS4):", t7)
        print("Testing Time (MMSE-CG3):", t8)
        print("Testing Time (MMSE-CG4):", t9)

    ber_MMSE = error_MMSE / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSENSA3 = error_MMSENSA3 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSENSA4 = error_MMSENSA4 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSESOR2 = error_MMSESOR2 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSESOR3 = error_MMSESOR3 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSEGS3 = error_MMSEGS3 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSEGS4 = error_MMSEGS4 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSECG3 = error_MMSECG3 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_MMSECG4 = error_MMSECG4 / (2 * TestDataLen * TxAntNum * bitperSym)


    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    p1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    p2 = plt.semilogy(SNRdB, ber_MMSENSA3, 'r--^', label='MMSE-NSA-k=3')
    p3 = plt.semilogy(SNRdB, ber_MMSENSA4, 'r--s', label='MMSE-NSA-k=4')
    p4 = plt.semilogy(SNRdB, ber_MMSESOR2, 'k-d', label='MMSE-SOR-k=2')
    p5 = plt.semilogy(SNRdB, ber_MMSESOR3, 'k-o', label='MMSE-SOR-k=3')
    p6 = plt.semilogy(SNRdB, ber_MMSEGS3, 'g-v', label='MMSE-GS-k=3')
    p7 = plt.semilogy(SNRdB, ber_MMSEGS4, 'g-D', label='MMSE-GS-k=4')
    p8 = plt.semilogy(SNRdB, ber_MMSECG3, 'y-v', label='MMSE-CG-k=3')
    p9 = plt.semilogy(SNRdB, ber_MMSECG4, 'y-D', label='MMSE-CG-k=4')
    plt.legend()
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title(str(RxAntNum) + r'$\times$' + str(TxAntNum) + ', MIMO, ' + str(Model) + ', Kron' +str(Kron))

    plt.show()
