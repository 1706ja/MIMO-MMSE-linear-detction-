import torch
import numpy as np
# import multiprocessing
# from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import cmath


# Parameters Settingh
TestDataLen = 4000
TxAntNum = 32  # Number of transmitting antennas
RxAntNum = 128 # Number of receiving antennas tested
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
error_MMSEw1NSA3 = np.zeros(SNRNum)
error_MMSEw1NSA4 = np.zeros(SNRNum)
error_MMSEw2NSA3 = np.zeros(SNRNum)
error_MMSEw2NSA4 = np.zeros(SNRNum)
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



def MMSE_NSA3test(x, y, H, Nv):
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
    IDE4 = IDE3 - np.matmul(DinvE**3,Dinv)

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

# weighted NSA
def MMSE_w1NSA3test(x, y, H, Nv):
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
    # k=4
    IDE4 = IDE3 - 0.5*np.matmul(DinvE**3,Dinv)

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

def MMSE_w1NSA4test(x, y, H, Nv):
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
    IDE2 = Dinv - np.matmul(DinvE,Dinv)
    # k=3
    IDE3 = IDE2 + 0.5*np.matmul(DinvE**2,Dinv)
    # k=4
    IDE4 = IDE3 - 0.5*np.matmul(DinvE**3,Dinv)

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
# weighted NSA (w = 0.75)
def MMSE_w2NSA3test(x, y, H, Nv):
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
    IDE3 = IDE2 + 0.75*np.matmul(DinvE**2,Dinv)
    # k=4
    IDE4 = IDE3 - 0.75*np.matmul(DinvE**3,Dinv)

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

def MMSE_w2NSA4test(x, y, H, Nv):
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
    IDE2 = Dinv - np.matmul(DinvE,Dinv)
    # k=3
    IDE3 = IDE2 + 0.75*np.matmul(DinvE**2,Dinv)
    # k=4
    IDE4 = IDE3 - 0.75*np.matmul(DinvE**3,Dinv)

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

if __name__ == "__main__":
    for nEN in range(SNRNum):
        print(SNRdB[nEN])
        for index in range (0,TestDataLen):
            x, y, H, Nv = GenerateTestData( TxAntNum, RxAntNum,
                                       DataLen, SNRdB[nEN], SNRdB[nEN])

            error_MMSE[nEN] += MMSEtest(x, y, H, Nv)

            error_MMSENSA3[nEN] += MMSE_NSA3test(x, y, H, Nv)

            error_MMSENSA4[nEN] += MMSE_NSA4test(x, y, H, Nv)

            error_MMSEw1NSA3[nEN] += MMSE_w1NSA3test(x, y, H, Nv)

            error_MMSEw1NSA4[nEN] += MMSE_w1NSA4test(x, y, H, Nv)

            error_MMSEw2NSA3[nEN] += MMSE_w2NSA3test(x, y, H, Nv)

            error_MMSEw2NSA4[nEN] += MMSE_w2NSA4test(x, y, H, Nv)

    ber_MMSE = error_MMSE / (TestDataLen * TxAntNum)
    ber_MMSENSA3 = error_MMSENSA3 / (TestDataLen * TxAntNum)
    ber_MMSENSA4 = error_MMSENSA4 / (TestDataLen * TxAntNum)
    ber_MMSEw1NSA3 = error_MMSEw1NSA3 / (TestDataLen * TxAntNum)
    ber_MMSEw1NSA4 = error_MMSEw1NSA4 / (TestDataLen * TxAntNum)
    ber_MMSEw2NSA3 = error_MMSEw2NSA3 / (TestDataLen * TxAntNum)
    ber_MMSEw2NSA4 = error_MMSEw2NSA4 / (TestDataLen * TxAntNum)



    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    p1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    p2 = plt.semilogy(SNRdB, ber_MMSENSA3, 'r--^', label='MMSE-NSA-k=3')
    p3 = plt.semilogy(SNRdB, ber_MMSENSA4, 'r--s', label='MMSE-NSA-k=4')
    p4 = plt.semilogy(SNRdB, ber_MMSEw1NSA3, 'k-d', label='MMSE-NSA-k=3,w=0.5')
    p5 = plt.semilogy(SNRdB, ber_MMSEw1NSA4, 'k-o', label='MMSE-NSA-k=4,w=0.5')
    p6 = plt.semilogy(SNRdB, ber_MMSEw2NSA3, 'g-v', label='MMSE-NSA-k=3,w=0.75')
    p7 = plt.semilogy(SNRdB, ber_MMSEw2NSA4, 'g-D', label='MMSE-NSA-k=4,w=0.75')
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


