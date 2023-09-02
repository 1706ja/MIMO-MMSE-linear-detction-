import torch
import numpy as np
# import multiprocessing
# from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import cmath


# Parameters Settingh
TestDataLen = 500000
TxAntNum = 16  # Number of transmitting antennas
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
error_MMSESOR2 = np.zeros(SNRNum)
error_MMSESOR3 = np.zeros(SNRNum)
error_MMSEp1SOR2 = np.zeros(SNRNum)
error_MMSEp1SOR3 = np.zeros(SNRNum)
error_MMSESSOR2 = np.zeros(SNRNum)
error_MMSESSOR3 = np.zeros(SNRNum)

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
    egn, eva = np.linalg.eig(np.matmul(Dinv, E))
    r = max(abs(egn))
    w = 2 / (1 + cmath.sqrt(1 - r ** 2))
    U = np.triu(E,0)
    L = E-U
    C = (D/w)+L

    C=  np.linalg.inv(C)

    f = 1/w*C

    C = np.matmul(C,(L-E/w))
    f = np.matmul(f,HTy)
    x1 = np.zeros(TxAntNum)
    # k=1
    x2 = np.matmul(C,x1)+f
    # k=2
    x3 = np.matmul(C,x2)+f
    # k = 3
    x4 = np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x3
    xhat = np.asarray(xhat)
    for index in range(0, TxAntNum):
        if xhat[index][0] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i][0] != x[i]:
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
    egn, eva = np.linalg.eig(np.matmul(Dinv, E))
    r = max(abs(egn))
    w = 2 / (1 + cmath.sqrt(1 - r ** 2))
    U = np.triu(E,0)
    L = E-U
    C = (D/w)+L

    C=  np.linalg.inv(C)

    f = 1/w*C

    C = np.matmul(C,(L-E/w))
    f = np.matmul(f,HTy)
    x1 = np.zeros(TxAntNum)
    # k=1
    x2 = np.matmul(C,x1)+f
    # k=2
    x3 = np.matmul(C,x2)+f
    # k = 3
    x4 = np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x4
    xhat = np.asarray(xhat)
    for index in range(0, TxAntNum):
        if xhat[index][0] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i][0] != x[i]:
            error_MMSESOR3 += 1
    return error_MMSESOR3


def MMSE_p1SOR2test(x, y, H, Nv):
    error_MMSEpSOR1 = 0

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
    # egn, eva = np.linalg.eig(np.matmul(Dinv,E))
    # r = max(abs(egn))
    # w = 2/(1+cmath.sqrt(1-r**2))
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
    # k=2
    x3 = np.matmul(C,x2)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x3

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEpSOR1 += 1
    return error_MMSEpSOR1

def MMSE_p1SOR3test(x, y, H, Nv):
    error_MMSEpSOR2 = 0

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
    # egn, eva = np.linalg.eig(np.matmul(Dinv, E))
    # r = max(abs(egn))
    # w = 2 / (1 + cmath.sqrt(1 - r ** 2))
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
    # k=2
    x3 = np.matmul(C,x2)+f
    # k = 3
    x4 = np.matmul(C,x3)+f


    # MIMO detection (MMSE) using perfect CSI

    xhat = x4

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEpSOR2 += 1
    return error_MMSEpSOR2

def MMSE_SSOR2test(x, y, H, Nv):
    error_MMSEpSOR1 = 0

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
    # egn, eva = np.linalg.eig(np.matmul(Dinv,E))
    # r = max(abs(egn))
    # w = 2/(1+cmath.sqrt(1-r**2))
    a = RxAntNum / (TxAntNum + Nv ** 2)
    b = 1 + cmath.sqrt(TxAntNum / RxAntNum)
    b = b ** 2
    w = 2 / (1 + cmath.sqrt(2 * a * b))
    U = np.triu(E,0)
    L = E-U
    C1 = D+w*L
    C1 = np.linalg.inv(C1)
    C1 = np.matmul(C1,(D+w*L-w*A))

    C2 = D + w*U
    C2 = np.linalg.inv(C2)
    C2 = np.matmul(C2, (D + w * U - w * A))

    f = w*HTy
    x1 = np.matmul(Dinv,HTy)
    f1 = np.matmul(C1,f)
    f2 = np.matmul(C2,f)
    x1 = np.matmul(Dinv,HTy)
    # k=1
    x2 = np.matmul(C1,x1)+f1
    x2 = np.matmul(C2,x2)+f2
    # k=2
    # x3 = np.matmul(C1,x2)+f1
    # x3 = np.matmul(C2,x3)+f2



    # MIMO detection (MMSE) using perfect CSI

    xhat = x2

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEpSOR1 += 1
    return error_MMSEpSOR1

def MMSE_SSOR3test(x, y, H, Nv):
    error_MMSEpSOR2 = 0

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
    # egn, eva = np.linalg.eig(np.matmul(Dinv, E))
    # r = max(abs(egn))
    # w = 2 / (1 + cmath.sqrt(1 - r ** 2))
    a = RxAntNum / (TxAntNum + Nv ** 2)
    b = 1 + cmath.sqrt(TxAntNum / RxAntNum)
    b = b ** 2
    w = 2 / (1 + cmath.sqrt(2 * a * b))
    U = np.triu(E,0)
    L = E-U
    C1 = D+w*L
    C1 = np.linalg.inv(C1)
    C1 = np.matmul(C1,(D+w*L-w*A))

    C2 = D + w*U
    C2 = np.linalg.inv(C2)
    C2 = np.matmul(C2, (D + w * U - w * A))

    f = w*HTy
    f1 = np.matmul(C1,f)
    f2 = np.matmul(C2,f)
    x1 = np.matmul(Dinv,HTy)
    # k=1
    x2 = np.matmul(C1,x1)+f1
    x2 = np.matmul(C2,x2)+f2
    # k=2
    x3 = np.matmul(C1,x2)+f1
    x3 = np.matmul(C2,x3)+f2
    #k=3
    # x4 = np.matmul(C1,x3)+f1
    # x4 = np.matmul(C2,x4)+f2


    # MIMO detection (MMSE) using perfect CSI

    xhat = x3

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSEpSOR2 += 1
    return error_MMSEpSOR2

if __name__ == "__main__":
    for nEN in range(SNRNum):
        print(SNRdB[nEN])
        for index in range (0,TestDataLen):
            x, y, H, Nv = GenerateTestData( TxAntNum, RxAntNum,
                                       DataLen, SNRdB[nEN], SNRdB[nEN])

            error_MMSE[nEN] += MMSEtest(x, y, H, Nv)

            error_MMSESOR2[nEN] += MMSE_SOR2test(x, y, H, Nv)

            error_MMSEp1SOR3[nEN] += MMSE_p1SOR3test(x, y, H, Nv)

            error_MMSEp1SOR2[nEN] += MMSE_p1SOR2test(x, y, H, Nv)

            error_MMSESSOR3[nEN] += MMSE_SSOR3test(x, y, H, Nv)

            error_MMSESSOR2[nEN] += MMSE_SSOR2test(x, y, H, Nv)


    ber_MMSE = error_MMSE / (TestDataLen * TxAntNum)
    ber_MMSESOR2 = error_MMSESOR2 / (TestDataLen * TxAntNum)
    ber_MMSESOR3 = error_MMSESOR3 / (TestDataLen * TxAntNum)
    ber_MMSEp1SOR2 = error_MMSEp1SOR2 / (TestDataLen * TxAntNum)
    ber_MMSEp1SOR3 = error_MMSEp1SOR3 / (TestDataLen * TxAntNum)
    ber_MMSESSOR2 = error_MMSESSOR2 / (TestDataLen * TxAntNum)
    ber_MMSESSOR3 = error_MMSESSOR3 / (TestDataLen * TxAntNum)
    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    p1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    p2 = plt.semilogy(SNRdB, ber_MMSESOR2, 'k-d', label='MMSE-SOR-k=2')
    p3 = plt.semilogy(SNRdB, ber_MMSESOR3, 'k-o', label='MMSE-SOR-k=3')
    p4 = plt.semilogy(SNRdB, ber_MMSEp1SOR2, 'r--^', label='MMSE-wappro-SOR-k=2')
    p5 = plt.semilogy(SNRdB, ber_MMSEp1SOR3, 'r--*', label='MMSE-wappro-SOR-k=3')
    p6 = plt.semilogy(SNRdB, ber_MMSESSOR2, 'g-D', label='MMSE-SSOR-k=2')
    p7 = plt.semilogy(SNRdB, ber_MMSESSOR3, 'g-H', label='MMSE-SSOR-k=3')
    plt.legend()
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title(str(RxAntNum) + r'$\times$' + str(TxAntNum) + ', MIMO, '+'16QAM')
    # if save_data==True:
    #     plt.savefig('MIMO Detection, '+str(RxAntNum)+'x'+str(TxAntNum)
    #                 +', Rayleigh Channel'+'.pdf',dpi=300,format='pdf')
    # PATH = './aprDiv.mat'
    # scio.savemat(PATH, {'SNRdB': SNRdB, 'ber_MMSE': ber_MMSE, 'ber_QMMSE': ber_QMMSE})
    plt.show()

