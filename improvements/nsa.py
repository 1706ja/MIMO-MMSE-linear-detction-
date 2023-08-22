import torch
import numpy as np
# import multiprocessing
# from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import cmath


# Parameters Settingh
TestDataLen = 100000
TxAntNum = 16  # Number of transmitting antennas
RxAntNum = 32 # Number of receiving antennas tested
DataLen = 1  # Length of data sequence
# MaxIter = 5  # Max iteration number of iterative algorithms

# SNR setting
SNRdBLow = 0  # Minimal value of SNR in dB
SNRdBHigh = 12  # Maximal value of SNR in dB
SNRIterval = 2  # Interval value of SNR sequence
SNRNum = int((SNRdBHigh - SNRdBLow ) / SNRIterval) +1  # Number of SNR sequence
SNRdB = np.linspace(SNRdBLow, SNRdBHigh, SNRNum)


# Variable Initialization
error_MMSE = np.zeros(SNRNum)
error_MMSENSA3 = np.zeros(SNRNum)
error_MMSENSA4 = np.zeros(SNRNum)
error_MMSEwNSA3 = np.zeros(SNRNum)
error_MMSEwNSA4 = np.zeros(SNRNum)
error_MMSEtNSA2 = np.zeros(SNRNum)
error_MMSEtNSA3 = np.zeros(SNRNum)
error_MMSElNSA2 = np.zeros(SNRNum)
error_MMSElNSA3 = np.zeros(SNRNum)

# Tri-diagonal Matrix Generation
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


def GenerateTestData(TxAntNum, RxAntNum, DataLen, SNRdBLow, SNRdBHigh):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


def MMSEtest(x, y, H, Nv):
    error_MMSE = 0
    #HTy = HT.matmul(y)
    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT,H)
    HTy = np.matmul(HT,y)
    Sigma = HTH
    o = Nv**2*np.identity(TxAntNum)
    Sigma = Sigma+o
    Sigma = np.linalg.inv(Sigma)
    xhat = np.matmul(Sigma, HTy)
    for index in range (0,TxAntNum):
        if xhat[index].real > 0:
            xhat[index] = 1
        else :
            xhat[index] = -1
    for i in range (0,TxAntNum):
        if xhat[i]!=x[i]:
            error_MMSE += 1
    return error_MMSE


# NSA iteration = 4
def MMSE_NSA3test(x, y, H, Nv):
    error_MMSENSA3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o

    v = np.diag(A)
    # Diagonal Matrix Generation
    D = np.diag(v)

    # Inverse of Diagonal Matrix
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

# Weighted NSA
# Uses The matrix w*D as Iterative Matrix Instead 
def MMSE_wNSA3test(x, y, H, Nv):
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
    IDE3 = IDE2 + 0.5*np.matmul(DinvE**2,Dinv)


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

def MMSE_wNSA4test(x, y, H, Nv):
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
    IDE3 = IDE2 + 0.5*np.matmul(DinvE ** 2, Dinv)
    # k=4
    IDE4 = IDE3 - 0.5*np.matmul(DinvE ** 3, Dinv)

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


# Tri-Diagonal NSA
# Uses The Tri-Diagonal Matrix as Iterative Matrix Instead 
def MMSE_tNSA2test(x, y, H, Nv):
    error_MMSENSA2 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    # print(H)
    # print('wwwwwwwwwwwwwwwwwwwwwwwww')
    # print(HT)

    v = np.diag(A)
    v1 = np.diag(A,-1)
    D = tridiag(v1,v,v1)


    Dinv = np.linalg.inv(D)

    E = A - D
    Dinv = np.asmatrix(Dinv)
    # k=2
    DinvE = np.matmul(Dinv,E)
    IDE2 = Dinv - np.matmul(DinvE,Dinv)


    Ainv_apr = IDE2

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
            error_MMSENSA2 += 1
    return error_MMSENSA2

def MMSE_tNSA3test(x, y, H, Nv):
    error_MMSENSA3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    # print(H)
    # print('wwwwwwwwwwwwwwwwwwwwwwwww')
    # print(HT)

    v = np.diag(A)
    v1 = np.diag(A,-1)
    D = tridiag(v1,v,v1)


    Dinv = np.linalg.inv(D)

    E = A - D
    Dinv = np.asmatrix(Dinv)
    # k=2
    DinvE = np.matmul(Dinv,E)
    IDE2 = Dinv - np.matmul(DinvE,Dinv)
    # k=3
    IDE3 = IDE2 + np.matmul(DinvE**2,Dinv)


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

def MMSE_lNSA2test(x, y, H, Nv):
    error_MMSENSA2 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    # print(H)
    # print('wwwwwwwwwwwwwwwwwwwwwwwww')
    # print(HT)

    D = np.tril(A)

    E = A - D
    Dinv = np.linalg.inv(D)
    # k=2
    DinvE = np.matmul(Dinv,E)
    IDE2 = Dinv - np.matmul(DinvE,Dinv)


    Ainv_apr = IDE2

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
            error_MMSENSA2 += 1
    return error_MMSENSA2

def MMSE_lNSA3test(x, y, H, Nv):
    error_MMSENSA3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)
    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    # print(H)
    # print('wwwwwwwwwwwwwwwwwwwwwwwww')
    # print(HT)

    D = np.tril(A)

    E = A - D
    Dinv = np.linalg.inv(D)
    # k=2
    DinvE = np.matmul(Dinv,E)
    IDE2 = Dinv - np.matmul(DinvE,Dinv)
    # k=3
    IDE3 = IDE2 + np.matmul(DinvE**2,Dinv)


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

if __name__ == "__main__":
    for nEN in range(SNRNum):
        print(SNRdB[nEN])
        for index in range (0,TestDataLen):
            x, y, H, Nv = GenerateTestData( TxAntNum, RxAntNum,
                                       DataLen, SNRdB[nEN], SNRdB[nEN])

            error_MMSE[nEN] += MMSEtest(x, y, H, Nv)

            error_MMSENSA3[nEN] += MMSE_NSA3test(x, y, H, Nv)

            error_MMSENSA4[nEN] += MMSE_NSA4test(x, y, H, Nv)

            error_MMSEwNSA3[nEN] += MMSE_wNSA3test(x, y, H, Nv)

            error_MMSEwNSA4[nEN] += MMSE_wNSA4test(x, y, H, Nv)

            error_MMSEtNSA2[nEN] += MMSE_tNSA2test(x, y, H, Nv)

            error_MMSEtNSA3[nEN] += MMSE_tNSA3test(x, y, H, Nv)

            error_MMSElNSA2[nEN] += MMSE_lNSA2test(x, y, H, Nv)

            error_MMSElNSA3[nEN] += MMSE_lNSA3test(x, y, H, Nv)

    ber_MMSE = error_MMSE / (TestDataLen * TxAntNum)
    ber_MMSENSA3 = error_MMSENSA3 / (TestDataLen * TxAntNum)
    ber_MMSENSA4 = error_MMSENSA4 / (TestDataLen * TxAntNum)
    ber_MMSEwNSA3 = error_MMSEwNSA3 / (TestDataLen * TxAntNum)
    ber_MMSEwNSA4 = error_MMSEwNSA4 / (TestDataLen * TxAntNum)
    ber_MMSEtNSA3 = error_MMSEtNSA2 / (TestDataLen * TxAntNum)
    ber_MMSEtNSA4 = error_MMSEtNSA3 / (TestDataLen * TxAntNum)
    ber_MMSElNSA3 = error_MMSElNSA2 / (TestDataLen * TxAntNum)
    ber_MMSElNSA4 = error_MMSElNSA3 / (TestDataLen * TxAntNum)



    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    p1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    p2 = plt.semilogy(SNRdB, ber_MMSENSA3, 'r--^', label='MMSE-NSA-k=3')
    p3 = plt.semilogy(SNRdB, ber_MMSENSA4, 'r--s', label='MMSE-NSA-k=4')
    p4 = plt.semilogy(SNRdB, ber_MMSEtNSA3, 'k-d', label='MMSE-triNSA-k=2')
    p5 = plt.semilogy(SNRdB, ber_MMSEtNSA4, 'k-o', label='MMSE-triNSA-k=3')
    p6 = plt.semilogy(SNRdB, ber_MMSElNSA3, 'g-v', label='MMSE-lowNSA-k=2')
    p7 = plt.semilogy(SNRdB, ber_MMSElNSA4, 'g-D', label='MMSE-lowNSA-k=3')
    p8 = plt.semilogy(SNRdB, ber_MMSEwNSA3, 'y-v', label='MMSE-weightedNSA-k=3')
    p9 = plt.semilogy(SNRdB, ber_MMSEwNSA4, 'y-D', label='MMSE-weightedNSA-k=4')
    # p4 = plt.semilogy(SNRdB, ber_MMSESOR2, 'k-d', label='MMSE-SOR-k=2')
    # p5 = plt.semilogy(SNRdB, ber_MMSESOR3, 'k-o', label='MMSE-SOR-k=3')
    # p6 = plt.semilogy(SNRdB, ber_MMSEGS3, 'g-v', label='MMSE-GS-k=3')#注意！注意！颜色！
    # p7 = plt.semilogy(SNRdB, ber_MMSEGS4, 'g-D', label='MMSE-GS-k=4')
    plt.legend()
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title(str(RxAntNum) + r'$\times$' + str(TxAntNum) + ', MIMO, ' + '16QAM')
    # if save_data==True:
    #     plt.savefig('MIMO Detection, '+str(RxAntNum)+'x'+str(TxAntNum)
    #                 +', Rayleigh Channel'+'.pdf',dpi=300,format='pdf')
    # PATH = './aprDiv.mat'
    # scio.savemat(PATH, {'SNRdB': SNRdB, 'ber_MMSE': ber_MMSE, 'ber_QMMSE': ber_QMMSE})
    plt.show()


