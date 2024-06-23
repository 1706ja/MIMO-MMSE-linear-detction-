import torch
import numpy as np
# import multiprocessing
#from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import tqdm
import cmath


# Parameters Settingh
TestDataLen = 8000
TxAntNum = 64  # Number of transmitting antennas
RxAntNum = 64 # Number of receiving antennas tested
DataLen = 1  # Length of data sequence
# MaxIter = 5  # Max iteration number of iterative algorithms

K1 = 1
K2 = 2
K3 = 4
K4 = 8
K5 = 16
# SNR setting
SNRdBLow = 10  # Minimal value of SNR in dB
SNRdBHigh = 50  # Maximal value of SNR in dB
SNRIterval = 4  # Interval value of SNR sequence
SNRNum = int((SNRdBHigh - SNRdBLow ) / SNRIterval) +1  # Number of SNR sequence
SNRdB = np.linspace(SNRdBLow, SNRdBHigh, SNRNum)

bitperSym = 1
ModType = 16
if ModType==2:
    Model = '4QAM'
    Cons = torch.tensor([-1., 1.])
    fnorm = 1/np.sqrt(2)
    bitCons = torch.tensor([[0],[1]])
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



KK           = [1,2,4,8,16]

error_KBEST = np.zeros((SNRNum,20))
# Variable Initialization
error_MMSE = np.zeros(SNRNum)
error_KBEST1 = np.zeros(SNRNum)
error_KBEST2 = np.zeros(SNRNum)
error_KBEST3 = np.zeros(SNRNum)
error_KBEST4 = np.zeros(SNRNum)
error_KBEST5 = np.zeros(SNRNum)

error_MMSENSA3 = np.zeros(SNRNum)
error_MMSENSA4 = np.zeros(SNRNum)
error_MMSESOR2 = np.zeros(SNRNum)
error_MMSESOR3 = np.zeros(SNRNum)
error_MMSEGS3 = np.zeros(SNRNum)
error_MMSEGS4 = np.zeros(SNRNum)
error_MMSECG3 = np.zeros(SNRNum)
error_MMSECG4 = np.zeros(SNRNum)

def GenerateTestData(TxAntNum, RxAntNum,  SNRdBLow, SNRdBHigh):
    x_ = np.random.randint(0,2,(TxAntNum,1))
    for i in range (0,TxAntNum):
        if x_[i] == 0:
            x_[i] = -1

    RandSNRdB = np.random.uniform(low=SNRdBLow, high=SNRdBHigh)
    SNR = 10 ** (RandSNRdB / 10)


    Nv_ = math.sqrt(TxAntNum/(2*SNR))
    H_=  np.sqrt(1/2)*(torch.randn(RxAntNum, TxAntNum) + 1j*torch.randn(RxAntNum, TxAntNum))
    H_ /= np.sqrt(torch.norm(H_) ** 2 / (2 * TxAntNum))
    H_=H_.numpy()
    y_ = np.matmul(H_,x_)

    n = Nv_ * torch.randn(RxAntNum, 1)*1/2
    n = n.numpy()
    y_ = y_+n
    return x_, y_, H_, Nv_

def GenerateTestData1(TxAntNum, RxAntNum, DataLen, SNRdBLow):

    # x_ = torch.zeros([2*TxAntNum, DataLen])
    # y_ = torch.zeros([2*RxAntNum, DataLen])
    # H_ = torch.zeros([2*RxAntNum, 2*TxAntNum])
    # Nv_ = 0
    # H = KronChannel(TxAntNum, RxAntNum, SampleNum, TSratio, RSratio)

    SNR = 10**(SNRdBLow/10)
    # Generate real and image part of data sequence seperately
    TxDataSyms = torch.randint(0, ModType, size=(2*TxAntNum, DataLen))
    # TxData_r, TxData_i = Modulation(TxDataBits, ModType, Cons)
    TxData_r = Cons[TxDataSyms[:TxAntNum, :]]
    TxData_i = Cons[TxDataSyms[TxAntNum:, :]]
    # Transform complex Tx signals to real
    x_ = torch.cat((TxData_r, TxData_i), dim=0)
    # x_[itr, :, :] = TxData
    # TxSymbol = np.concatenate((TxPilot, TxData), 1)
    # Generate channel matrix (Rayleigh channel)
    Hc = np.sqrt(1/2)*(torch.randn(RxAntNum, TxAntNum) + 1j*torch.randn(RxAntNum, TxAntNum))
    # Hc = H[itr]
    # Transform complex channle matrix to real
    HMat = torch.cat((torch.cat((torch.real(Hc), -torch.imag(Hc)), 1),
                          torch.cat((torch.imag(Hc), torch.real(Hc)), 1)), 0).float()
    # Normalize the column of real channel matrix
    HMat /= np.sqrt(torch.norm(HMat)**2/(2*TxAntNum))
    # Data send via the channels without AWGN
    RxSymbol_noAWGN = torch.matmul(HMat, x_)
    # Calculate the norm of channel matrix
    Hnorm = torch.norm(HMat)**2
    # Noise variance & adding AWGN
    Nv_ = (1*Hnorm)/(2*SNR*(2*RxAntNum)*fnorm**2)
    y_ = RxSymbol_noAWGN + np.sqrt(Nv_)*torch.randn(2*RxAntNum, DataLen)
    H_ = HMat
    # y_[itr, :, :] = RxSymbol
    # H_[itr, :, :] = Hhat
    # Nv_[itr, :, :] = Nv


    return x_, y_, H_, Nv_

def GenerateTestData2(H, y):

    H1 = np.asmatrix(H)
    y1 = np.asmatrix(y)
    Q,R = np.linalg.qr(H)
    y_ = np.matmul(Q.T,y1)
    return R, y1

def K_Best(H, Cons, y, K, x):
    error_Kbest = 0
    H = np.asmatrix(H)
    y = np.asmatrix(y)
    Cons = np.asmatrix(Cons)
    Q,R = np.linalg.qr(H)
    Hy = np.matmul(Q.T,y)
    len_y = len(Hy)
    Cons_size = np.size(Cons)
    curSIV = np.zeros((len_y, K))  # Current Selected Input Vectors
    curPED = np.zeros((1,K))  # Current PED values
    nextPED = np.zeros((1,Cons_size * K))  # Extended selected PED values
    nextCons = np.zeros((1,Cons_size * K))  # Extended each PED value corresponding to the position map
    a2 = R[-1,-1]*Cons - Hy[-1]
    a= np.array(a2)
    temp = a * a
    # print(temp)
    # print(temp[0])
    temp = temp[0]
    index = np.argsort(temp)
    temp = np.sort(temp)
    curSIV[-1, :] = Cons[0,index[:K]]
    curPED = temp[:K]
    # nextCons = copy(Cons,K,Cons_size)
    for k in range (K):
        nextCons[0, k * Cons_size: (k + 1) * Cons_size] = Cons

    for n in range(len_y - 2, -1, -1):
        for k in range(K):
            a = np.dot(R[n, :], curSIV[:, k])

            b = R[n, n] * Cons + a[0,0] - Hy[n]
            # print(b)
            # print(np.multiply(b,b))
            # print('*'*50)
            # print(nextPED[0, k * Cons_size: (k + 1) * Cons_size])
            # print(curPED)
            # print(curPED[k])
            nextPED[0, k * Cons_size: (k + 1) * Cons_size] = np.multiply(b,b)  + curPED[k]
        index2 = np.argsort(nextPED)
        index2 = index2[0]
        nextPED = np.sort(nextPED)
        pos = np.floor(index2[:K] / Cons_size).astype(int)
        curSIV = curSIV[:, pos]
        # print(curSIV)
        curSIV[n, :] = np.copy(nextCons[0,index2[:K]])
        curPED = np.copy(nextPED[0,:K])


    Opt = curSIV[:, 0]
    # print(Opt)
    Opt = torch.tensor(Opt)
    Opt = torch.reshape(Opt,(len_y,1))
    _, indices_x = torch.min((x - Cons) ** 2, dim=-1, keepdim=True)
    _, indices = torch.min((Opt - Cons) ** 2, dim=-1, keepdim=True)
    Rxbit = bitCons[indices]
    xbit = bitCons[indices_x]
    comp = torch.where(
        Rxbit != xbit)  # return tuple (Tensor, Tensor, Tensor), where contains the 3D coordinates of non-zero elements
    error_Kbest += len(comp[0])
    # error_Kbest += len(comp[0])

    return error_Kbest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


def MMSEtest(x, y, H, Nv):
    error_MMSE = 0

    HTy = H.T.matmul(y)
    HTH = H.T.matmul(H)

    # MIMO detection (MMSE) using perfect CSI
    Sigma = torch.inverse(HTH + Nv * fnorm ** 2 * torch.unsqueeze(torch.eye(2 * TxAntNum), 0))
    xhat = torch.matmul(Sigma, HTy)
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

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o

    v = np.diag(A)
    D = np.diag(v)

    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range (0,TxAntNum):
        u_r[index] = v[index].real/abs(v[index])**2
        u_i[index] = v[index].imag/abs(v[index])**2
    u = u_r+1j*u_i
    Dinv = np.diag(u)

    E = A - D
    Dinv = np.asmatrix(Dinv)
    # k=2
    DinvE = np.matmul(Dinv,E)
    IDE2 = Dinv - np.matmul(DinvE,Dinv)
    # k=3
    IDE3 = IDE2 + np.matmul(DinvE**2,Dinv)
    # k=4
    Ainv_apr = IDE3

    # MIMO detection (MMSE) using perfect CSI
    Sigma = Ainv_apr
    xhat = np.matmul(Sigma, HTy)
    xhat = np.array(xhat)
    for index in range (0,TxAntNum):
        if xhat[index]>0 :
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range (0,TxAntNum):
        if xhat[i]!=x[i]:
            error_MMSENSA3 += 1
    return error_MMSENSA3

def MMSE_NSA4test(x, y, H, Nv):
    error_MMSENSA4 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)

    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)

    E = A - D
    Dinv = np.asmatrix(Dinv)
    C = A*Dinv

    # k=2
    DinvE = np.matmul(Dinv, E)
    IDE2 = Dinv - np.matmul(DinvE, Dinv)
    # k=3
    IDE3 = IDE2 + np.matmul(DinvE ** 2, Dinv)
    # k=4
    IDE4 = IDE3 - np.matmul(DinvE ** 3, Dinv)

    Ainv_apr = IDE4

    # MIMO detection (MMSE) using perfect CSI
    Sigma = Ainv_apr
    xhat = np.matmul(Sigma, HTy)
    xhat = np.array(xhat)

    for index in range (0,TxAntNum):
        if xhat[index]>0 :
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range (0,TxAntNum):
        if xhat[i]!=x[i]:
            error_MMSENSA4 += 1
    return error_MMSENSA4

def MMSE_SOR2test(x, y, H, Nv):
    error_MMSESOR2 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)
    E = A-D
    a = RxAntNum / (TxAntNum + Nv ** 2)
    b = 1 + cmath.sqrt(TxAntNum / RxAntNum)
    b = b ** 2
    w = 2 / (1 + cmath.sqrt(2 * a * b))
    U = np.triu(E,0)
    L = E-U
    C = (D/w)+L
    C=  np.linalg.inv(C)
    f = 1/w*C
    C = np.matmul(C,(L-E/w))
    f = np.matmul(f,HTy)
    x1 = np.matmul(Dinv,HTy)
    # k=1
    x2 = np.matmul(C,x1)+f
    # # k=2
    # x3 = np.matmul(C,x2)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x2

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSESOR2 += 1
    return error_MMSESOR2

def MMSE_SOR3test(x, y, H, Nv):
    error_MMSESOR3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)

    E = A-D

    a = RxAntNum / (TxAntNum + Nv ** 2)
    b = 1 + cmath.sqrt(TxAntNum / RxAntNum)
    b = b ** 2
    w = 2 / (1 + cmath.sqrt(2 * a * b))
    # although the above computation of w
    # is the most sufficient, it does not
    # match the need of MIMO-detection
    # an alternative way is basically
    # setting w as a certain number
    # w = 1.5
    U = np.triu(E,0)
    L = E-U
    C = (D/w)+L
    C=  np.linalg.inv(C)
    f = 1/w*C
    C = np.matmul(C,(L-E/w))
    f = np.matmul(f,HTy)
    x1 = np.matmul(Dinv,HTy)
    # k=1
    x2 = np.matmul(C,x1)+f
    # k=2
    x3 = np.matmul(C,x2)+f
    # k = 3
    # x4 = np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x3

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSESOR3 += 1
    return error_MMSESOR3

def MMSE_GS4test(x, y, H, Nv):
    error_MMSECG3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)
    E = A - D
    U = np.triu(E, 0)
    L = E - U
    C = np.linalg.inv(D+L)
    f = np.matmul(C,HTy)
    C = np.matmul(C,U)
    x1 = np.matmul(Dinv,HTy)
    # k=1
    x2 = -np.matmul(C,x1)+f
    # k=2
    x3 = -np.matmul(C,x2)+f
    # k =3
    x4 = -np.matmul(C,x3)+f
    #k = 4
    # x5 = -np.matmul(C,x4)+f

    # MIMO detection (MMSE) using perfect CSI

    xhat = x3
    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG3 += 1

    return error_MMSECG3

def MMSE_GS3test(x, y, H, Nv):
    error_MMSECG3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)
    E = A - D
    U = np.triu(E, 0)
    L = E - U
    C = np.linalg.inv(D+L)
    f = np.matmul(C,HTy)
    C = np.matmul(C,U)
    x1 = np.matmul(Dinv,HTy)
    # k=1
    x2 = -np.matmul(C,x1)+f
    # k=2
    x3 = -np.matmul(C,x2)+f
    # # k =3
    # x4 = -np.matmul(C,x3)+f

    xhat = x2
    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG3 += 1

    return error_MMSECG3

def MMSE_CG3test(x, y, H, Nv):
    error_MMSECG3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)
    xhat = np.matmul(Dinv,HTy)
    r = HTy - np.matmul(A, xhat)
    p = r
    #k = 1
    alpha1 = np.matmul(r.H,r)
    alpha2 = np.matmul(p.H,A)
    alpha2 = np.matmul(alpha2,p)
    alpha = alpha1/alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat+alpha*p
    beta1 = np.matmul(r.H,r)
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H,r)
    beta = beta2/beta1
    beta = beta.tolist()
    beta = beta[0][0]
    p = r+beta*p
    # k=2
    alpha1 = np.matmul(r.H,r)
    alpha2 = np.matmul(p.H,A)
    alpha2 = np.matmul(alpha2,p)
    alpha = alpha1/alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat+alpha*p
    beta1 = np.matmul(r.H,r)
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H,r)
    beta = beta2/beta1
    beta = beta.tolist()
    beta = beta[0][0]
    p = r+beta*p
    # #k=3
    # alpha1 = np.matmul(r.H,r)
    # alpha2 = np.matmul(p.H,A)
    # alpha2 = np.matmul(alpha2,p)
    # alpha = alpha1/alpha2
    # alpha = alpha.tolist()
    # alpha = alpha[0][0]
    # xhat = xhat+alpha*p
    # beta1 = np.matmul(r.H,r)
    # r -= alpha * np.matmul(A, p)
    # beta2 = np.matmul(r.H,r)
    # beta = beta2/beta1
    # beta = beta.tolist()
    # beta = beta[0][0]
    # p = r+beta*p
    # MIMO detection (MMSE) using perfect CSI


    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG3 += 1

    return error_MMSECG3

def MMSE_CG4test(x, y, H, Nv):
    error_MMSECG4 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    D = np.diag(v)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)
    xhat = np.matmul(Dinv,HTy)
    #k=1
    r = HTy - np.matmul(A, xhat)
    p = r

    alpha1 = np.matmul(r.H, r)
    alpha2 = np.matmul(p.H, A)
    alpha2 = np.matmul(alpha2, p)
    alpha = alpha1 / alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat + alpha * p
    beta1 = np.matmul(r.H, r)
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H, r)
    beta = beta2/beta1
    beta = beta.tolist()
    beta = beta[0][0]
    p = r + beta * p
    # k=2

    alpha1 = np.matmul(r.H, r)
    alpha2 = np.matmul(p.H, A)
    alpha2 = np.matmul(alpha2, p)
    alpha = alpha1 / alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat + alpha * p
    beta1 = np.matmul(r.H, r)
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H, r)
    beta = beta2/beta1
    beta = beta.tolist()
    beta = beta[0][0]
    p = r + beta * p
    # k=3

    alpha1 = np.matmul(r.H, r)
    alpha2 = np.matmul(p.H, A)
    alpha2 = np.matmul(alpha2, p)
    alpha = alpha1 / alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat + alpha * p
    beta1 = np.matmul(r.H, r)
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H, r)
    beta = beta2/beta1
    beta = beta.tolist()
    beta = beta[0][0]
    p = r + beta * p
    # # k=4
    #
    # alpha1 = np.matmul(r.H, r)
    # alpha2 = np.matmul(p.H, A)
    # alpha2 = np.matmul(alpha2, p)
    # alpha = alpha1 / alpha2
    # alpha = alpha.tolist()
    # alpha = alpha[0][0]
    # xhat = xhat + alpha * p
    # beta1 = np.matmul(r.H, r)
    # r -= alpha * np.matmul(A, p)
    # beta2 = np.matmul(r.H, r)
    # beta = beta2/beta1
    # beta = beta.tolist()
    # beta = beta[0][0]
    # p = r + beta * p
    # MIMO detection (MMSE) using perfect CSI


    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG4 += 1

    return error_MMSECG4

if __name__ == "__main__":
    for nEN in range(SNRNum):
        # print(SNRdB[nEN])
        for index in tqdm.tqdm(range (0,TestDataLen)):
            x, y, H, Nv = GenerateTestData1( TxAntNum, RxAntNum, DataLen,
                                       SNRdB[nEN])
            # R, Hy = GenerateTestData2(H,y)
            error_MMSE[nEN] += MMSEtest(x, y, H, Nv)
            error_KBEST1[nEN] += K_Best(H,Cons,y,K1,x)

            error_KBEST2[nEN] += K_Best(H, Cons, y, K2, x)

            error_KBEST3[nEN] += K_Best(H, Cons, y, K3, x)

            error_KBEST4[nEN] += K_Best(H, Cons, y, K4, x)

            error_KBEST5[nEN] += K_Best(H, Cons, y, K5, x)

            # error_MMSENSA3[nEN] += MMSE_NSA3test(x, y, H, Nv)
            #
            # error_MMSENSA4[nEN] += MMSE_NSA4test(x, y, H, Nv)
            #
            # error_MMSESOR2[nEN] += MMSE_SOR2test(x, y, H, Nv)
            #
            # error_MMSESOR3[nEN] += MMSE_SOR3test(x, y, H, Nv)
            #
            # error_MMSEGS3[nEN] += MMSE_GS3test(x, y, H, Nv)
            #
            #
            # error_MMSEGS4[nEN] += MMSE_GS4test(x, y, H, Nv)
            #
            # error_MMSECG3[nEN] += MMSE_CG3test(x, y, H, Nv)
            #
            # error_MMSECG4[nEN] += MMSE_CG4test(x, y, H, Nv)


    ber_MMSE = error_MMSE / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_KBEST = error_KBEST1 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_KBEST2 = error_KBEST2 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_KBEST3 = error_KBEST3 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_KBEST4 = error_KBEST4 / (2 * TestDataLen * TxAntNum * bitperSym)
    ber_KBEST5 = error_KBEST5 / (2 * TestDataLen * TxAntNum * bitperSym)
    # ber_KBESTK = error_KBEST / (2 * TestDataLen * TxAntNum * bitperSym)
    # ber_MMSENSA3 = error_MMSENSA3 / (TestDataLen * TxAntNum)
    # ber_MMSENSA4 = error_MMSENSA4 / (TestDataLen * TxAntNum)
    # ber_MMSESOR2 = error_MMSESOR2 / (TestDataLen * TxAntNum)
    # ber_MMSESOR3 = error_MMSESOR3 / (TestDataLen * TxAntNum)
    # ber_MMSEGS3 = error_MMSEGS3 / (TestDataLen * TxAntNum)
    # ber_MMSEGS4 = error_MMSEGS4 / (TestDataLen * TxAntNum)
    # ber_MMSECG3 = error_MMSECG3 / (TestDataLen * TxAntNum)
    # ber_MMSECG4 = error_MMSECG4 / (TestDataLen * TxAntNum)

    print(ber_KBEST)
    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    P1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    P2 = plt.semilogy(SNRdB, ber_KBEST, 'r--^', label='KBEST, K=' + str(K1))
    P3 = plt.semilogy(SNRdB, ber_KBEST2, 'k-d', label='KBEST, K=' + str(K2))
    P4 = plt.semilogy(SNRdB, ber_KBEST3, 'c--s', label='KBEST, K=' + str(K3))
    P5 = plt.semilogy(SNRdB, ber_KBEST4, 'g-v', label='KBEST, K=' + str(K4))
    P6 = plt.semilogy(SNRdB, ber_KBEST5, 'y-D', label='KBEST, K=' + str(K5))
    # p3 = plt.semilogy(SNRdB, ber_MMSENSA4, 'r--s', label='MMSE-NSA-k=4')
    # p4 = plt.semilogy(SNRdB, ber_MMSESOR2, 'k-d', label='MMSE-SOR-k=2')
    # p5 = plt.semilogy(SNRdB, ber_MMSESOR3, 'k-o', label='MMSE-SOR-k=3')
    # p6 = plt.semilogy(SNRdB, ber_MMSEGS3, 'g-v', label='MMSE-GS-k=3')
    # p7 = plt.semilogy(SNRdB, ber_MMSEGS4, 'g-D', label='MMSE-GS-k=4')
    # p8 = plt.semilogy(SNRdB, ber_MMSECG3, 'y-v', label='MMSE-CG-k=3')#注意！注意！颜色！
    # p9 = plt.semilogy(SNRdB, ber_MMSECG4, 'y-D', label='MMSE-CG-k=4')
    plt.legend()
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title(str(TxAntNum) + r'$\times$' + str(RxAntNum) + ', MIMO ,' + str(Model))
    # if save_data==True:
    #     plt.savefig('MIMO Detection, '+str(RxAntNum)+'x'+str(TxAntNum)
    #                 +', Rayleigh Channel'+'.pdf',dpi=300,format='pdf')
    # PATH = './aprDiv.mat'
    # scio.savemat(PATH, {'SNRdB': SNRdB, 'ber_MMSE': ber_MMSE, 'ber_QMMSE': ber_QMMSE})
    plt.show()
