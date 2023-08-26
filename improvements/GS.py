import torch
import numpy as np
# import multiprocessing
# from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import cmath


# Parameters Settingh
TestDataLen = 200000
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
error_MMSEGS3 = np.zeros(SNRNum)
error_MMSEGS4 = np.zeros(SNRNum)
error_MMSEtGS2 = np.zeros(SNRNum)
error_MMSEtGS3 = np.zeros(SNRNum)
error_MMSEnGS2 = np.zeros(SNRNum)
error_MMSEnGS3 = np.zeros(SNRNum)

def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

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



def MMSE_GS3test(x, y, H, Nv):
    error_MMSEGS3 = 0

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
    # x3 = -np.matmul(C,x2)+f
    # # k =3
    # x4 = -np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x2
    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEGS3 += 1

    return error_MMSEGS3

def MMSE_GS4test(x, y, H, Nv):
    error_MMSEGS3 = 0

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
    # x4 = -np.matmul(C,x3)+f
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
            error_MMSEGS3 += 1

    return error_MMSEGS3

def MMSE_tGS2test(x, y, H, Nv):
    error_MMSEGS3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    v1 = np.diag(A, -1)
    D = tridiag(v1, v, v1)
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


    # MIMO detection (MMSE) using perfect CSI

    xhat = x2
    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEGS3 += 1

    return error_MMSEGS3

def MMSE_tGS3test(x, y, H, Nv):
    error_MMSEGS3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    v = np.diag(A)
    v1 = np.diag(A, -1)
    D = tridiag(v1, v, v1)
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
            error_MMSEGS3 += 1

    return error_MMSEGS3

def MMSE_nGS2test(x, y, H, Nv):
    error_MMSEGS3 = 0

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
    DinvE = np.matmul(Dinv,E)
    L = E - U
    C = np.linalg.inv(D+L)
    f = np.matmul(C,HTy)
    C = np.matmul(C,U)
    S = Dinv - np.matmul(DinvE,Dinv)
    x1 = np.matmul(S,HTy)
    # k=1
    x2 = -np.matmul(C,x1)+f
    # k=2
    # x3 = -np.matmul(C,x2)+f
    # # k =3
    # x4 = -np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x2
    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEGS3 += 1

    return error_MMSEGS3

def MMSE_nGS3test(x, y, H, Nv):
    error_MMSEGS3 = 0

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
    DinvE = np.matmul(Dinv,E)
    L = E - U
    C = np.linalg.inv(D+L)
    f = np.matmul(C,HTy)
    C = np.matmul(C,U)
    S = Dinv - np.matmul(DinvE,Dinv)
    x1 = np.matmul(S,HTy)
    # k=1
    x2 = -np.matmul(C,x1)+f
    # k=2
    x3 = -np.matmul(C,x2)+f
    # # k =3
    # x4 = -np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x3
    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEGS3 += 1

    return error_MMSEGS3

if __name__ == "__main__":
    for nEN in range(SNRNum):
        print(SNRdB[nEN])
        for index in range (0,TestDataLen):
            x, y, H, Nv = GenerateTestData( TxAntNum, RxAntNum,
                                       SNRdB[nEN], SNRdB[nEN])

            error_MMSE[nEN] += MMSEtest(x, y, H, Nv)

            error_MMSEGS3[nEN] += MMSE_GS3test(x, y, H, Nv)

            error_MMSEGS4[nEN] += MMSE_GS4test(x, y, H, Nv)

            error_MMSEtGS2[nEN] += MMSE_tGS2test(x, y, H, Nv)

            error_MMSEtGS3[nEN] += MMSE_tGS3test(x, y, H, Nv)

            error_MMSEnGS2[nEN] += MMSE_nGS2test(x, y, H, Nv)

            error_MMSEnGS3[nEN] += MMSE_nGS3test(x, y, H, Nv)




    ber_MMSE = error_MMSE / (TestDataLen * TxAntNum)
    ber_MMSEGS3 = error_MMSEGS3 / (TestDataLen * TxAntNum)
    ber_MMSEGS4 = error_MMSEGS4 / (TestDataLen * TxAntNum)
    ber_MMSEtGS2 = error_MMSEtGS2 / (TestDataLen * TxAntNum)
    ber_MMSEtGS3 = error_MMSEtGS3 / (TestDataLen * TxAntNum)
    ber_MMSEnGS2 = error_MMSEnGS2 / (TestDataLen * TxAntNum)
    ber_MMSEnGS3 = error_MMSEnGS3 / (TestDataLen * TxAntNum)
    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    p1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    # p2 = plt.semilogy(SNRdB, ber_MMSENSA3, 'r--^', label='MMSE-NSA-k=3')
    # p3 = plt.semilogy(SNRdB, ber_MMSENSA4, 'r--s', label='MMSE-NSA-k=4')
    # p4 = plt.semilogy(SNRdB, ber_MMSESOR2, 'k-d', label='MMSE-SOR-k=2')
    # p5 = plt.semilogy(SNRdB, ber_MMSESOR3, 'k-o', label='MMSE-SOR-k=3')
    p2 = plt.semilogy(SNRdB, ber_MMSEGS3, 'g-v', label='MMSE-GS-k=3')
    p3 = plt.semilogy(SNRdB, ber_MMSEGS4, 'g-D', label='MMSE-GS-k=4')
    p4 = plt.semilogy(SNRdB, ber_MMSEtGS2, 'k-d', label='MMSE-triGS-k=2')
    p5 = plt.semilogy(SNRdB, ber_MMSEtGS3, 'k-o', label='MMSE-triGS-k=3')
    p6 = plt.semilogy(SNRdB, ber_MMSEnGS2, 'r--^', label='MMSE-iniGS-k=2')
    p7 = plt.semilogy(SNRdB, ber_MMSEnGS3, 'r--s', label='MMSE-iniGS-k=3')
    # p8 = plt.semilogy(SNRdB, ber_MMSEGS3, 'y-v', label='MMSE-GS-k=3')#注意！注意！颜色！
    # p9 = plt.semilogy(SNRdB, ber_MMSEGS4, 'y-D', label='MMSE-GS-k=4')
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
