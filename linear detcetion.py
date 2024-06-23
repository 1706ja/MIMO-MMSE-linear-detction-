import torch
import numpy as np
#import multiprocessing
#from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import cmath

# Parameters Settingh
TestDataLen = 500000
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
        print(SNRdB[nEN])
        for index in range (0,TestDataLen):
            x, y, H, Nv = GenerateTestData( TxAntNum, RxAntNum,
                                       SNRdB[nEN], SNRdB[nEN])

            error_MMSE[nEN] += MMSEtest(x, y, H, Nv)

            error_MMSENSA3[nEN] += MMSE_NSA3test(x, y, H, Nv)

            error_MMSENSA4[nEN] += MMSE_NSA4test(x, y, H, Nv)

            error_MMSESOR2[nEN] += MMSE_SOR2test(x, y, H, Nv)

            error_MMSESOR3[nEN] += MMSE_SOR3test(x, y, H, Nv)

            error_MMSEGS3[nEN] += MMSE_GS3test(x, y, H, Nv)


            error_MMSEGS4[nEN] += MMSE_GS4test(x, y, H, Nv)

            error_MMSECG3[nEN] += MMSE_CG3test(x, y, H, Nv)

            error_MMSECG4[nEN] += MMSE_CG4test(x, y, H, Nv)


    ber_MMSE = error_MMSE / (TestDataLen * TxAntNum)
    ber_MMSENSA3 = error_MMSENSA3 / (TestDataLen * TxAntNum)
    ber_MMSENSA4 = error_MMSENSA4 / (TestDataLen * TxAntNum)
    ber_MMSESOR2 = error_MMSESOR2 / (TestDataLen * TxAntNum)
    ber_MMSESOR3 = error_MMSESOR3 / (TestDataLen * TxAntNum)
    ber_MMSEGS3 = error_MMSEGS3 / (TestDataLen * TxAntNum)
    ber_MMSEGS4 = error_MMSEGS4 / (TestDataLen * TxAntNum)
    ber_MMSECG3 = error_MMSECG3 / (TestDataLen * TxAntNum)
    ber_MMSECG4 = error_MMSECG4 / (TestDataLen * TxAntNum)

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
    p8 = plt.semilogy(SNRdB, ber_MMSECG3, 'y-v', label='MMSE-CG-k=3')#注意！注意！颜色！
    p9 = plt.semilogy(SNRdB, ber_MMSECG4, 'y-D', label='MMSE-CG-k=4')
    plt.legend()
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title(str(RxAntNum) + r'$\times$' + str(TxAntNum) + ', MIMO' )
    # if save_data==True:
    #     plt.savefig('MIMO Detection, '+str(RxAntNum)+'x'+str(TxAntNum)
    #                 +', Rayleigh Channel'+'.pdf',dpi=300,format='pdf')
    # PATH = './aprDiv.mat'
    # scio.savemat(PATH, {'SNRdB': SNRdB, 'ber_MMSE': ber_MMSE, 'ber_QMMSE': ber_QMMSE})
    plt.show()
