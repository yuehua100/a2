import matplotlib.pyplot as plt
import json
import os
from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy import linalg as LA
from scipy.stats import kstest, chi2, normaltest, kurtosis
import scipy.stats as stats
import multiprocessing as mp
from itertools import repeat, combinations, groupby
import math
from hmmlearn import hmm
from tqdm import tqdm

def load_trajectory_from_folder(folder_path):
    trajectories = []
    modes = []
    tra_folder_path = os.path.join(folder_path, 'traj')

    # 搜索文件名
    for filename in natsorted(os.listdir(tra_folder_path)):
        # 确认轨迹文件
        if filename.endswith('.json'):
            # �����������ļ�·��
            file_path = os.path.join(tra_folder_path, filename)

            # 打开文件
            with open(file_path, 'r') as file:
                data = json.load(file)
            trajectories.extend(data)
            
            length = len(data)
            print(length)
            prefix = os.path.splittext(filename)[0]
            print(prefix)
            modes.extend([prefix]*length)

    return trajectories, modes


def get_Squaredis(x0,y0, x1, y1):
    return (x1-x0) ** 2 + (y1 - y0) ** 2

def MSD(x,y):
    r2 = []
    r4 = []
    N = len(x)
    for i in range(1,len(x)):
        x1 = np.array(x[:N-i])
        x2 = np.array(x[i:N])
        y1 = np.array(y[:N-i])
        y2 = np.array(y[i:N])
        c = (x2 - x1) ** 2 + (y2 - y1) ** 2
        r = np.mean(c)
        r1 = np.mean(c ** 2)
        r2.append(r)
        r4.append(r1)
    return r2,r4

def diff(msd,dt,frac):
    N = int(np.floor(frac*len(msd)))
    Fitmsd = msd[:N]
    n_list = np.arange(1,N+1)*dt
    popt,_ = curve_fit(lambda x, d, alpha: 4*d*(x)**alpha, n_list, Fitmsd, p0 = [Fitmsd[0]/(4*dt),1], bounds = [[0.00000001,0],[np.inf,5]])

    premsd = np.array([4 * popt[0] * (x)**popt[1] for x in n_list])
    res = Fitmsd-premsd

    popt, _ = curve_fit(lambda x, d, alpha: 4 * d * (x)**alpha, n_list, Fitmsd, p0=[Fitmsd[0]/(4*dt), 1], sigma=np.repeat(np.std(res, ddof=1), len(Fitmsd)), bounds=([0.00000001, 0], [np.inf, 5]))

    Chival = res**2/np.var(res,ddof=1)
    Pval = stats.chi2.sf(np.sum(Chival), len(Fitmsd)-len(popt))
    return popt[0],popt[1],Pval

def get_alpha(msd,dt,frac):
    N = int(np.floor(frac*len(msd)))
    Fitmsd = msd[:N]
    n_list = np.arange(1,N+1)*dt
    try:
        popt,_ = curve_fit(lambda x, logD, a: np.log(4) + logD + a * x, np.log(n_list), np.log(Fitmsd), bounds=[[-np.inf, 0], [np.inf, 3]])
        alpha = popt[1]
    except ValueError:
        alpha = 0
    return alpha

def get_alpha2_noise1(msd,dt,frac):
    N = int(np.floor(frac*len(msd)))
    Fitmsd = msd[:N]
    log_msd = np.log(Fitmsd)
    log_n = np.log(np.arange(1,N+1))
    alpha2 = (N * np.sum(log_n * log_msd) - np.sum(log_n * np.sum(log_msd))) / (N * np.sum(log_n ** 2) - (np.sum(log_n)) ** 2)

    return alpha2

def get_alpha3_noise2(msd,dt,frac):
    N = int(np.floor(frac*len(msd)))
    Fitmsd = msd[:N]
    n_list = np.arange(1,N+1)*dt

    s2_max = msd[0]
    s2_0 = msd[0]/2.0
    D_0 = msd[0]/(4*dt)
    eps = 0.0001

    try:
        popt,_ = curve_fit(lambda x, d, a, s2: 4 * d * (x) ** a + s2 ** 2, n_list, Fitmsd, p0 = (D_0,1,s2_0), bounds=[[0,0,0],[np.inf,5,s2_max]], ftol = eps)
        alpha3 = popt[1]
    except ValueError:
        alpha3 = 0

    return alpha3

def get_alpha4_noise3(msd,dt,frac):
    N = int(np.floor(frac*len(msd)))
    Fitmsd = msd[:N]
    n_list = np.arange(1,N+1)*dt

    D_0 = msd[0]/(4*dt)
    eps = 0.01

    try:
        popt,_ = curve_fit(lambda x, d, a: 4 * d * (x) ** a - 4 * d * (dt) ** a, n_list, Fitmsd, p0 = (D_0,1), bounds=[[0,0],[np.inf,5]], ftol = eps)
        alpha4 = popt[1]
    except ValueError:
        alpha4 = 0

    return alpha4

def get_MSDRatio(msd):
    msd = np.array(msd)
    n1 = np.arange(1,len(msd))
    n2 = np.arange(2,len(msd)+1)
    r_n1 = msd[0:len(msd)-1]
    r_n2 = msd[1:len(msd)]
    ratio = np.mean(r_n1 / r_n2 - n1 / n2)

    return ratio

def get_Gaussianity(msd,msd2):
    msd = np.array(msd)
    msd2 = np.array(msd2)

    Gn = np.mean(msd2/(2 * msd ** 2)) - 1
    return Gn

def get_tensor(x,y):
    mxx = np.mean((np.array(x) - np.mean(np.array(x))) ** 2)
    myy = np.mean((np.array(y) - np.mean(np.array(y))) ** 2)
    mxy = np.mean((np.array(x) - np.mean(np.array(x))) * (np.array(y) - np.mean(np.array(y))))

    tensors = np.array([[mxx, mxy], [mxy, myy]])
    return tensors

def get_Asymmetry(tensor):

    eigenvalues, eigenvectors = np.linalg.eig(tensor)
    lamda1 = eigenvalues[0]
    lamda2 = eigenvalues[1]

    asy = ((lamda1-lamda2) ** 2) / (2 * (lamda1+lamda2) ** 2)
    A = -1 * math.log(1-asy)

    return A

def get_Kurtosis(tensor,x,y):

    eigenvalues, eigenvectors = np.linalg.eig(tensor)
    index = np.argmax(np.abs(eigenvalues))

    dominant = eigenvectors[:,index]
    xp = np.array([np.dot(dominant, v) for v in np.array([x,y]).T])
    K = kurtosis(xp)
    return K

def get_Efficiency(x,y):
    N = len(x)
    Ee = get_Squaredis(x[0],y[0],x[-1],y[-1])
    Ed = sum([get_Squaredis(x[i],y[i],x[i+1],y[i+1]) for i in range(0, N-1)])
    eff2 = np.log(Ee / ((N-1) * Ed))

    return eff2.real

def get_Displacement(x,y):
    N = len(x)
    dis = np.sqrt(np.array([get_Squaredis(x[i],y[i],x[i+1],y[i+1]) for i in range(0, N-1)]))
    return dis

def get_Straightness(x,y,Dis):
    REe = np.sqrt(get_Squaredis(x[0],y[0],x[-1],y[-1]))
    REd = sum(Dis)
    S = REe / REd
    return S

def get_VACF(x,y):
    Cv = []
    lag = 1
    x = np.array(x)
    y = np.array(y)
    dis_x = x[lag:] - x[:-lag]
    dis_y = y[lag:] - y[:-lag]
    N = len(dis_x)
    for n in range(0,11):
        vx1 = np.array(dis_x[:N-n])
        vx2 = np.array(dis_x[n:])
        vy1 = np.array(dis_y[:N-n])
        vy2 = np.array(dis_y[n:])

        cv = vx1 * vx2 + vy1 * vy2
        cvv = np.mean(cv)

        Cv.append(cvv)

    Cv = Cv / Cv[0]

    return Cv[1], Cv[2], Cv[-1]

def get_Maxd(x,y):
    Trajs = np.array([x,y]).T
    subsets = list(combinations(Trajs, 2))
    Distance = [get_Squaredis(pair[0][0],pair[0][1],pair[1][0],pair[1][1]) for pair in subsets]
    maxd = np.sqrt(max(Distance))

    return maxd

def get_fractal(Dis,maxd):
    Ns = len(Dis)
    L = sum(Dis)
    Df = np.log(Ns) / (np.log(Ns * maxd / L))
    return Df

def get_Trappedness(r2,maxd):
    Dif = (r2[1]-r2[0])/1
    P = 1 - np.exp(0.2048 - 0.25117 * ((Dif * len(r2))/(maxd ** 2)))
    CP = P if P > 0 else 0
    return CP

def get_MaxExcursion(x,y,Dis):
    RMe = np.sqrt(get_Squaredis(x[0],y[0],x[-1],y[-1]))
    RMd = max(Dis)
    ME = RMd / RMe
    return ME

def get_Mean_MaxExcursion(x,y,Dis):
    N = len(x)
    R = np.array([get_Squaredis(x[0],y[0],x[i],y[i]) for i in range(1,N)])
    Rmax = np.sqrt(np.max(R))
    MME = Rmax/np.sqrt(sum(Dis ** 2)/2)

    return MME

def get_Pvariation(x,y):
    N = len(x)
    maxm = int(max(0.01 * N, 5))
    m_list = np.arange(1,maxm+1)
    p_list = np.arange(1,6)

    pvar = np.zeros((len(p_list), len(m_list)))

    for i in range(len(p_list)):
        for j in range(len(m_list)):
            p = p_list[i]
            m = m_list[j]
            vm = np.sqrt(np.array([get_Squaredis(x[k],y[k],x[k+m],y[k+m]) for k in range(0, N-m, m)]))
            vmp = sum(vm ** p)
            pvar[i][j] = vmp

    p_values = []
    for i in range(len(p_list)):
        pv = np.log(pvar[i])
        try:
            popt,_ = curve_fit(lambda x, a, b: a * (x) + b, np.log(m_list), pv)
            test_value = popt[0]
        except ValueError:
            test_value = (pv[-1]-pv[0])/(np.log(m_list[-1])-np.log(m_list[0]))
        p_values.append(test_value)

    return p_values

def get_Pvariation_feature(x,y):
    N = len(x)
    p_list = np.array([1/H for H in np.arange(0.1,1.01,0.1)])
    m_list = np.arange(1,16)

    pvar = np.zeros((len(p_list), len(m_list)))

    for i in range(len(p_list)):
        for j in range(len(m_list)):
            p = p_list[i]
            m = m_list[j]
            vm = np.sqrt(np.array([get_Squaredis(x[k],y[k],x[k+m],y[k+m]) for k in range(0, N-m, m)]))
            vmp = sum(vm ** p)
            pvar[i][j] = vmp

    p_values = []

    for i in range(len(p_list)):
        pv = np.log(pvar[i])
        try:
            popt,_ = curve_fit(lambda x, a, b: a * (x) + b, np.log(m_list), pv)
            test_value = popt[0]
        except ValueError:
            test_value = (pv[-1]-pv[0])/(np.log(m_list[-1])-np.log(m_list[0]))
        p_values.append(test_value)

    sign_pvar = np.nonzero(np.diff([np.sign(val) for val in p_values]))

    if len(sign_pvar) > 0:
        p_var_info = np.sign(p_values[0])
    else:
        p_var_info = 0

    return p_var_info

def get_DA_test(x,y):
    kx, px = normaltest(np.diff(np.array(x)))
    ky, py = normaltest(np.diff(np.array(y)))

    return kx, ky

def get_KS_test(x,y):
    dx = np.diff(np.array(x))
    dxn = (dx - np.nanmean(dx))/np.nanstd(dx)
    distpx = dxn ** 2

    dy = np.diff(np.array(y))
    dyn = (dy - np.nanmean(dy))/np.nanstd(dy)
    distpy = dyn ** 2

    distp = distpx + distpy

    ts = np.linspace(min(distp),max(distp),len(distp))
    ts_nonzero = np.where(ts==0, np.finfo(float).eps, ts)

    [stat,pv] = kstest(distp, 'chi2', args=(ts_nonzero,2), alternative='two-sided', mode='exact')

    return stat

def get_detrending_moving_average(x,y):
    Tmax = [1, 2]
    dma = []
    for i in Tmax:
        Alldisp_x = np.array([x[j:j+i+1] for j in range(len(x)-i)])
        Alldisp_y = np.array([y[j:j+i+1] for j in range(len(y)-i)])

        R2 = np.mean((Alldisp_x[:,-1] - np.mean(Alldisp_x, axis=1)) ** 2 + (Alldisp_y[:,-1] - np.mean(Alldisp_y, axis=1)) ** 2)
        dma.append(R2)

    return dma[0], dma[1]

def get_moving_windows(x,y):
    N = len(x)
    mwxy_value = []
    windows = [10, 20]

    xmean_value = []
    ymean_value = []
    xstd_value = []
    ystd_value = []
    
    for window in windows:
        while len(x) < window+3:
            window = len(x)-3
        Alldisp_x = np.array([x[j:j+window+1] for j in range(len(x)-window)])
        Alldisp_y = np.array([y[j:j+window+1] for j in range(len(y)-window)])
        mvw_x_std = np.std(Alldisp_x, axis=1)
        xstd = np.mean(np.abs(np.diff(np.sign(np.diff(mvw_x_std)))))/2
        mvw_x_mean = np.mean(Alldisp_x, axis=1)
        xmean = np.mean(np.abs(np.diff(np.sign(np.diff(mvw_x_mean)))))/2

        mvw_y_std = np.std(Alldisp_y, axis=1)
        ystd = np.mean(np.abs(np.diff(np.sign(np.diff(mvw_y_std)))))/2
        mvw_y_mean = np.mean(Alldisp_y, axis=1)
        ymean = np.mean(np.abs(np.diff(np.sign(np.diff(mvw_y_mean)))))/2

        xmean_value.append(xmean)
        ymean_value.append(ymean)
        xstd_value.append(xstd)
        ystd_value.append(ystd)
    
    mwxy_value = xmean_value+ymean_value+xstd_value+ystd_value
    return mwxy_value

def get_max_std(x,y):
    window = 3
    Alldisp_x = np.array([x[j:j+window] for j in range(len(x)-window)])
    Alldisp_y = np.array([y[j:j+window] for j in range(len(y)-window)])
    mvw_x_std = np.std(Alldisp_x, axis=1)
    mvw_y_std = np.std(Alldisp_y, axis=1)

    minx = np.nanmin(mvw_x_std)
    miny = np.nanmin(mvw_y_std)

    cor_minx = minx if minx > 1e-9 else 1e-9
    cor_miny = miny if miny > 1e-9 else 1e-9

    ratio_x = np.nanmax(mvw_x_std)/cor_minx
    ratio_y = np.nanmax(mvw_y_std)/cor_miny

    stdx = np.nanstd(np.array(x))
    stdy = np.nanstd(np.array(y))

    cor_stdx = stdx if stdx > 1e-10 else 1e-10
    cor_stdy = stdy if stdx > 1e-10 else 1e-10

    change_x = np.nanmax(np.abs(np.diff(mvw_x_std)))/cor_stdx
    change_y = np.nanmax(np.abs(np.diff(mvw_y_std)))/cor_stdy

    return ratio_x, ratio_y, change_x, change_y


def get_NMJ_exponents(x,y):
    start = 3
    Nmax = int(len(x)/2)
    s_list = np.arange(start,Nmax+1)

    NS = len(s_list)

    RSx = []
    RSy = []
    Mx = []
    My = []
    Nx = []
    Ny = []

    for i in s_list:
        Alldisp_x = [x[j:j+i] for j in range(0, len(x), i)]
        if len(Alldisp_x[-1]) < i:
            Alldisp_x = Alldisp_x[:-1]

        Alldisp_y = [y[j:j+i] for j in range(0, len(y), i)]
        if len(Alldisp_y[-1]) < i:
            Alldisp_y = Alldisp_y[:-1]

        deltax = [np.diff(disp) for disp in Alldisp_x]
        deltay = [np.diff(disp) for disp in Alldisp_y]
        Ytx = np.median([np.sum(np.abs(dx)) for dx in deltax])
        Yty = np.median([np.sum(np.abs(dy)) for dy in deltay])

        Mx.append(Ytx)
        My.append(Yty)

        Ztx = np.median([np.sum(dx**2) for dx in deltax])
        Zty = np.median([np.sum(dy**2) for dy in deltay])
        Nx.append(Ztx)
        Ny.append(Zty)

        temp_RStx = np.array([np.cumsum(dx - np.mean(dx)) for dx in deltax])
        max_temp_RStx = np.max(temp_RStx, axis=1)
        min_temp_RStx = np.min(temp_RStx, axis=1)
        stdx = [np.std(dx) for dx in deltax]
        stdx = [max(s, 1e-10) for s in stdx]
        RStx = [(max_temp_RStx[j] - min_temp_RStx[j]) / stdx[j] for j in range(len(deltax))]

        temp_RSty = np.array([np.cumsum(dy - np.mean(dy)) for dy in deltay])
        max_temp_RSty = np.max(temp_RSty, axis=1)
        min_temp_RSty = np.min(temp_RSty, axis=1)
        stdy = [np.std(dy) for dy in deltay]
        stdy = [max(s, 1e-10) for s in stdy]
        RSty = [(max_temp_RSty[j] - min_temp_RSty[j]) / stdy[j] for j in range(len(deltay))]

        RSx.append(np.mean(RStx))
        RSy.append(np.mean(RSty))

    s_list_log = np.log(s_list-1)
    if 0. in RSx:
        try:
            popt_rx, _ = curve_fit(lambda x, a, d, c: a*(x)**d + c, s_list-1, RSx)
            JX = popt_rx[1]
        except ValueError:
            JX = 0
    else:
        try:
            RSx_log = np.log(RSx)
            popt_rx, _ = curve_fit(lambda x, d, c: d*x + c, s_list_log, RSx_log)
            JX = popt_rx[0]
        except ValueError:
            JX = 0

    if 0. in RSy:
        try:
            popt_ry, _ = curve_fit(lambda x, a, d, c: a*(x)**d + c, s_list-1, RSy)
            JY = popt_ry[1]
        except ValueError:
            JY = 0
    else:
        try:
            RSy_log = np.log(RSy)
            popt_ry, _ = curve_fit(lambda x, d, c: d*x + c, s_list_log, RSy_log)
            JY = popt_ry[0]
        except ValueError:
            JY = 0

    popt_mx, _ = curve_fit(lambda x, d, c: d*x + c, s_list_log, np.log(Mx))
    MX = popt_mx[0]-0.5

    popt_my, _ = curve_fit(lambda x, d, c: d*x + c, s_list_log, np.log(My))
    MY = popt_my[0]-0.5

    popt_nx, _ = curve_fit(lambda x, d, c: d*x + c, s_list_log, np.log(Nx))
    NX = (popt_nx[0]+1-2*MX)/2

    popt_ny, _ = curve_fit(lambda x, d, c: d*x + c, s_list_log, np.log(Ny))
    NY = (popt_ny[0]+1-2*MY)/2

    return (JX+JY)/2, (MX+MY)/2, (NX+NY)/2

def get_power_spectral(x,y):
    Lt = len(x)
    spec = []
    f_list = [i/Lt for i in range(1,Lt)]
    for f in f_list:
        S2 = np.sum([np.exp(1j * f * k ) * (x[k] + y[k]) for k in range(0, Lt)])
        S1 = np.abs(S2) ** 2 / (Lt-1)
        spec.append(S1)
    popt_spec, _ = curve_fit(lambda x, d, c: d*x + c, np.log(f_list), np.log(spec))
    ps = popt_spec[0]
    return ps

def HMM(x,y):
    probs = [];
    x = np.array(x)
    y = np.array(y)
    dis_x = x[1:] - x[:-1]
    dis_y = y[1:] - y[:-1]
    dis = np.array([dis_x, dis_y]).T
    
    model_1_state = hmm.GaussianHMM(n_components=1, covariance_type="full", n_iter=1000)
    model_2_states = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)

    model_1_state.fit(dis)
    model_2_states.fit(dis)

    state = model_1_state.predict(dis)
    states = model_2_states.predict(dis)
    
    log_likelihood_1 = model_1_state.score(dis)
    probs.append(log_likelihood_1)
    log_likelihood_2 = model_2_states.score(dis)
    probs.append(log_likelihood_2)
    
    aic_1 = model_1_state.aic(dis)
    probs.append(aic_1)
    aic_2 = model_2_states.aic(dis)
    probs.append(aic_2)
    
    bic_1 = model_1_state.bic(dis)
    probs.append(bic_1)
    bic_2 = model_2_states.bic(dis)
    probs.append(bic_2)
    
    return state, model_1_state, states, model_2_states, probs

def Time_in(model, states):
    #Disd = model.means_
    #norms = np.linalg.norm(Disd, axis=1)
    #Effd = [i for i in norms]
    Var = model.covars_
    Effd = [(Var[i,0,0]+Var[i,1,1])/2 for i in range(2)]
    
    statemap = dict(zip(np.arange(2)[np.argsort(Effd)], np.arange(2)))
    newstates = [statemap[s] for s in states]
    counts = [newstates.count(s)/len(newstates) for s in range(2)]

    arr = np.array(newstates)
    lengths = [(key, len(list(group))) for key, group in groupby(arr)]
    one_lengths = [length for value, length in lengths if value==1.0]
    zero_lengths = [length for value, length in lengths if value==0.0]

    lifetime_0 = sum(zero_lengths)/len(zero_lengths)
    lifetime_1 = sum(one_lengths)/len(one_lengths)

    min_0 = min(zero_lengths)
    min_1 = min(one_lengths)
    
    return counts[0], counts[1], lifetime_0, lifetime_1, min_0, min_1

def get_features(x,y,frac_max,dt):

    msd, msd2 = MSD(x,y)
    D, alpha, Pval = diff(msd,dt,frac_max)
    alpha = get_alpha(msd,dt,frac_max)
    alpha2 = get_alpha2_noise1(msd,dt,frac_max)
    alpha3 = get_alpha3_noise2(msd,dt,frac_max)
    alpha4 = get_alpha4_noise3(msd,dt,frac_max)
    MSD_ratio = get_MSDRatio(msd)
    nonGauss = get_Gaussianity(msd,msd2)
    tensor = get_tensor(x,y)
    asymmetry = get_Asymmetry(tensor)
    kurt = get_Kurtosis(tensor,x,y)
    eff = get_Efficiency(x,y)
    displacements = get_Displacement(x,y)
    straightness = get_Straightness(x,y,displacements)
    vacf1, vacf2, vacfend = get_VACF(x,y)
    maxd = get_Maxd(x,y)
    fractal = get_fractal(displacements,maxd)
    trappness = get_Trappedness(msd, maxd)
    max_excursion = get_MaxExcursion(x,y,displacements)
    mean_max_excursion = get_Mean_MaxExcursion(x,y,displacements)
    Pvalues = get_Pvariation(x,y)
    Pfeature_val = get_Pvariation_feature(x,y)
    DAx, DAy = get_DA_test(x,y)
    KStest = get_KS_test(x,y)
    dma1, dma2 = get_detrending_moving_average(x,y)
    mwxy_values = get_moving_windows(x,y)
    ratiox, ratioy, changex, changey = get_max_std(x,y)
    J, M, L = get_NMJ_exponents(x,y)
    power_spec = get_power_spectral(x,y)
    
    #state, model_1, states, model_2, probs = HMM(x,y)
    #partial_1, partial_2, duration_1, duration_2, min_dur_1, min_dur_2 = Time_in(model_2, states)

    feature_values = [alpha, D, Pval, alpha2, alpha3, alpha4, MSD_ratio, nonGauss, asymmetry, kurt, eff, straightness, vacf1, vacf2, vacfend, fractal, trappness, max_excursion , mean_max_excursion]\
                    +list(Pvalues)+[Pfeature_val, DAx, DAy, KStest, dma1, dma2] + mwxy_values + [ratiox, ratioy, changex, changey, J, M, L, power_spec]
    return feature_values

feature_names = ["alpha", "diff", "pval", "alpha_2", "alpha_3", "alpha_4", "MSD_ratio", "Gaussianity", "asymmetry", "kurtosis", "efficiency", \
                "straightness", "VAF1", "VAF2", "VAF10", "Fractal", "Trappedness", \
                "max_excursion", "mean_max_excur", "P_variation_1", "P_variation_2", \
                "P_variation_3", "P_variation_4", "P_variation_5", "P_feature_var", \
                "dagostino_stats_x", "dagostino_stats_y", "ksstat_chi2", "dma1", \
                "dma2", "mwmean_x_10", "mwmean_x_20", "mwmean_y_10", "mwmean_y_20", \
                "mwstd_x_10", "mwstd_x_20", "mwstd_y_10", "mwstd_y_20", "max_std_x", \
                "max_std_y", "max_std_change_x", "max_std_change_y", "J", "M", "L", \
                "Power_spectral"]

# ��ȡ�켣
"""folder_path = '/data/hbc/sklearn/KG'
traces, modes = load_trajectory_from_folder(folder_path)"""
folder_path = '/data/zyh'

file_path = "/home/gwswdp02/data/zyh/trajs/BW.json"
with open(file_path, 'r') as file:
    traces = json.load(file)

length = len(traces)
print(length)
modes = ['BW']*length

dt = 0.05
frac_max = 0.5
min_traj_length = 20
pixel_to_micro = 1.0

filter_traces = [trace for trace in traces if len(trace) >= min_traj_length]

allfeatures = []

for t in tqdm(filter_traces):
    t_array = np.array(t)
    x_array, y_array = t_array[:, 0]*pixel_to_micro, t_array[:, 1]*pixel_to_micro
    x, y = x_array.tolist(), y_array.tolist()
    features = get_features(x,y,frac_max,dt)
    allfeatures.append(features)

finaldata = pd.DataFrame(allfeatures, columns=feature_names)

output_feature_file = os.path.join(folder_path, 'feature', 'features_zyh_BW.csv')
finaldata.to_csv(output_feature_file,index=False)

Mode = pd.DataFrame(modes, columns=["Mode"]) 
output_mode_file = os.path.join(folder_path, 'feature', 'mode_name_zyh_BW.csv')
Mode.to_csv(output_mode_file,index=False)
