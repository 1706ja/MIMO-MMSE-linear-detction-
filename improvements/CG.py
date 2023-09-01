import torch
import numpy as np
# import multiprocessing
# from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import cmath


# Parameters Settingh
TestDataLen = 80000
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
error_MMSECG3 = np.zeros(SNRNum)
error_MMSECG4 = np.zeros(SNRNum)
error_MMSEpCG3 = np.zeros(SNRNum)
error_MMSEpCG4 = np.zeros(SNRNum)
error_MMSEsCG3 = np.zeros(SNRNum)
error_MMSEsCG4 = np.zeros(SNRNum)
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

# conjugate-gradient, itr = 3
def MMSE_CG3test(x, y, H, Nv):
    error_MMSECG3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    xhat = 0*HTy
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
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H,r)
    beta = beta2/alpha1
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
    #k=3
    alpha1 = np.matmul(r.H,r)
    alpha2 = np.matmul(p.H,A)
    alpha2 = np.matmul(alpha2,p)
    alpha = alpha1/alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat+alpha*p


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
    xhat = 0*HTy
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
    r -= alpha * np.matmul(A, p)
    beta2 = np.matmul(r.H,r)
    beta = beta2/alpha1
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
    #k=3
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
    #k=4
    alpha1 = np.matmul(r.H,r)
    alpha2 = np.matmul(p.H,A)
    alpha2 = np.matmul(alpha2,p)
    alpha = alpha1/alpha2
    alpha = alpha.tolist()
    alpha = alpha[0][0]
    xhat = xhat+alpha*p


    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG4 += 1

    return error_MMSECG4

# pre-computing conjugate-gradient, itr = 3
def MMSE_pCG3test(x, y, H, Nv):
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
    xhat = np.matmul(Dinv, HTy)
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



    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG4 += 1

    return error_MMSECG4

def MMSE_pCG4test(x, y, H, Nv):
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







    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG4 += 1

    return error_MMSECG4


# split-CG, itr = 3
def MMSE_sCG3test(x, y, H, Nv):
    error_MMSECG3 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    A1 = A.tolist()
    B = 0*np.identity(TxAntNum) + 0*1j * np.identity(TxAntNum)
    for i in range(0, TxAntNum):
            B[i][i] = 1 / cmath.sqrt(A1[i][i])
    xhat = 0*HTy
    y1 = np.matmul(B,HTy)
    p1 = y1

    #k=1
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)
    y1 = y1 - a*p2
    a2 = np.matmul(y1.H,y1)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    p1 = y1+a*p1
    #k=2
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)
    y1 = y1 - a*p2
    a2 = np.matmul(y1.H,y1)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    p1 = y1+a*p1
    #k=3
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)

    for index in range(0, TxAntNum):
        if xhat[index] > 0:
            xhat[index] = 1
        else:
            xhat[index] = -1
    for i in range(0, TxAntNum):
        if xhat[i] != x[i]:
            error_MMSECG3 += 1

    return error_MMSECG3


def MMSE_sCG4test(x, y, H, Nv):
    error_MMSECG4 = 0

    H = np.asmatrix(H)
    HT = H.H
    HTH = np.matmul(HT, H)
    HTy = np.matmul(HT, y)

    o = Nv ** 2 * np.identity(TxAntNum)
    A = HTH + o
    A1 = A.tolist()
    B = 0*np.identity(TxAntNum) + 0*1j * np.identity(TxAntNum)
    for i in range(0, TxAntNum):
            B[i][i] = 1 / cmath.sqrt(A1[i][i])
    v = np.diag(A)
    u_r = np.zeros(TxAntNum)
    u_i = np.zeros(TxAntNum)
    for index in range(0, TxAntNum):
        u_r[index] = v[index].real / abs(v[index]) ** 2
        u_i[index] = v[index].imag / abs(v[index]) ** 2
    u = u_r + 1j * u_i
    Dinv = np.diag(u)
    xhat = np.matmul(Dinv,HTy)
    y1 = HTy - np.matmul(A,xhat)
    p1 = y1

    #k=1
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)
    y1 = y1 - a*p2
    a2 = np.matmul(y1.H,y1)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    p1 = y1+a*p1
    #k=2
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)
    y1 = y1 - a*p2
    a2 = np.matmul(y1.H,y1)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    p1 = y1+a*p1
    #k=3
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)
    y1 = y1 - a*p2
    a2 = np.matmul(y1.H,y1)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    p1 = y1+a*p1
    #k=4
    p2 = np.matmul(B,A)
    p2 = np.matmul(p2,B)
    p2 = np.matmul(p2,p1)
    a1 = np.matmul(y1.H,y1)
    a2 = np.matmul(p1.H,p2)
    a = a1/a2
    a = a.tolist()
    a = a[0][0]
    xhat = xhat+ a*np.matmul(B,p1)


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

            error_MMSECG3[nEN] += MMSE_CG3test(x, y, H, Nv)

            error_MMSECG4[nEN] += MMSE_CG4test(x, y, H, Nv)

            error_MMSEpCG3[nEN] += MMSE_pCG3test(x, y, H, Nv)

            error_MMSEpCG4[nEN] += MMSE_pCG4test(x, y, H, Nv)

            error_MMSEsCG3[nEN] += MMSE_sCG3test(x, y, H, Nv)

            error_MMSEsCG4[nEN] += MMSE_sCG4test(x, y, H, Nv)


    ber_MMSE = error_MMSE / (TestDataLen * TxAntNum)
    ber_MMSECG3 = error_MMSECG3 / (TestDataLen * TxAntNum)
    ber_MMSECG4 = error_MMSECG4 / (TestDataLen * TxAntNum)
    ber_MMSEpCG3 = error_MMSEpCG3 / (TestDataLen * TxAntNum)
    ber_MMSEpCG4 = error_MMSEpCG4 / (TestDataLen * TxAntNum)
    ber_MMSEsCG3 = error_MMSEsCG3 / (TestDataLen * TxAntNum)
    ber_MMSEsCG4 = error_MMSEsCG4 / (TestDataLen * TxAntNum)

    save_data = False

    plt.figure(1)
    # plt.style.use("_classic_test_patch")
    p1 = plt.semilogy(SNRdB, ber_MMSE, 'b-o', label='MMSE')
    # p2 = plt.semilogy(SNRdB, ber_MMSENSA3, 'r--^', label='MMSE-NSA-k=3')
    # p3 = plt.semilogy(SNRdB, ber_MMSENSA4, 'r--s', label='MMSE-NSA-k=4')
    # p4 = plt.semilogy(SNRdB, ber_MMSESOR2, 'k-d', label='MMSE-SOR-k=2,w=c')
    # p5 = plt.semilogy(SNRdB, ber_MMSESOR3, 'k-o', label='MMSE-SOR-k=3,w=c')
    # p6 = plt.semilogy(SNRdB, ber_MMSEGS2, 'g-v', label='MMSE-GS-k=2,ini')
    # p7 = plt.semilogy(SNRdB, ber_MMSEGS3, 'g-D', label='MMSE-GS-k=3,ini')
    p2 = plt.semilogy(SNRdB, ber_MMSECG3, 'y-v', label='MMSE-CG-k=3')#注意！注意！颜色！
    p3 = plt.semilogy(SNRdB, ber_MMSECG4, 'y-D', label='MMSE-CG-k=4')
    p4 = plt.semilogy(SNRdB, ber_MMSEpCG3, 'r--s', label='MMSE-preCG-k=2')#注意！注意！颜色！
    p5 = plt.semilogy(SNRdB, ber_MMSEpCG4, 'r--*', label='MMSE-preCG-k=3')
    p6 = plt.semilogy(SNRdB, ber_MMSEsCG3, 'g-v', label='MMSE-splitCG-k=3')#注意！注意！颜色！
    p7 = plt.semilogy(SNRdB, ber_MMSEsCG4, 'g-D', label='MMSE-splitCG-k=4')
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
