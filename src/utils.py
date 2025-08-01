# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import zmq
import pickle
import numpy as np
import os.path as op
from configobj import ConfigObj
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import random
import os
import obspy
from obspy import Stream
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import correlate, find_peaks

rstate = np.random.RandomState(333)


class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods
    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype, shape).
    """
    def send_array(self, arr, flags=0, copy=True, track=False):
        md = dict(
            dtype=str(arr.dtype),
            shape=arr.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(arr, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        arr = np.frombuffer(msg, dtype=md['dtype'])
        return arr.reshape(md['shape'])


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


def string_decode(section):
    keywords = ['station', 'savepath']

    for key in section:
        if key in keywords:
            continue
        try:
            section[key] = eval(section[key])
        except:
            for i, value in enumerate(section[key]):
                section[key][i] = eval(value)
    return section


def load_params(initfile):
    config = ConfigObj(initfile)
    keywords = ['station', 'savepath']
    params = []
    for configsection in config.sections:
        if configsection == 'datapaths':
            continue
        section = config[configsection]
        section = string_decode(section)
        params.append(section)
    return params


def load_params_user(initfile, station, slowness=7):
    import linecache
    config = ConfigObj(initfile)

    paths = {}
    for key in config['datapaths']:
        if key.split('.')[-1] == 'bin':
            file = config['datapaths'][key] % (station, slowness)
        else:
            file = config['datapaths'][key] % station

        if op.exists(file):
            newkey = key.split('_')[-1]
            paths[newkey] = file

            # only for receiver functions
            if key.split('.')[-1] == 'bin':
                slow = float(linecache.getline(file, 2).strip().replace('#', ''))
                paths['slowness.bin'] = slow

            if key.split('.')[-1] == 'stack':
                slow = float(linecache.getline(file, 2).strip().replace('#', ''))
                paths['slowness.stack'] = slow

    modelpriors = string_decode(config['modelpriors'])
    initparams = string_decode(config['initparams'])
    initparams['station'] = station
    initparams['savepath'] = initparams['savepath'] % (station, '%.2f')
    return paths, modelpriors, initparams


def save_baywatch_config(targets, path='.', priors=dict(), initparams=dict(),
                         refmodel=dict()):
    """
    Saves a configfile that you will need if using BayWatch.

    targets: JointTarget instance fed into the inversion
    path: where to save the configfile
    priors: used for inversion
    refmodel: reference model / values to be illustrated
    """
    configfile = op.join(path, 'baywatch.pkl')
    data = {}

    for target in targets.targets:
        target.get_covariance = None

    data['targets'] = targets.targets
    data['priors'] = priors
    data['initparams'] = initparams
    data['refmodel'] = refmodel

    with open(configfile, 'wb') as f:
        pickle.dump(data, f)


def save_config(targets, configfile, priors=dict(), initparams=dict()):
    """
    Conveniently saves a configfile that you can easily use to view the data
    and parameters used for inversion. This configfile (.pkl) will also be used
    for PlotFromStorage plotting methods after the inversion. With this you can
    redo the plots with the correct data used for the inversion.

    targets: JointTarget instance from inversion
    configfile: outfile name
    priors, initparams: parameter dictionaries important for plotting,
    contains e.g. prior distributions, noise params, iterations etc.
    """
    data = {}
    refs = []

    for target in targets.targets:
        target.get_covariance = None
        ref = target.ref
        refs.append(ref)

    data['targets'] = targets.targets
    data['targetrefs'] = refs
    data['priors'] = priors
    data['initparams'] = initparams

    with open(configfile, 'wb') as f:
        pickle.dump(data, f)


def read_config(configfile):
    try:  # python2
        with open(configfile, 'rb') as f:
            data = pickle.load(f)
    except:  # python3
        with open(configfile, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data


def get_path(name):
    fn = op.join(op.dirname(__file__), 'defaults', name)
    if not op.exists(fn):
        raise OSError('%s does not exist!' % name)
    return fn


# following functions for estimate r_RF
def _compute_gaussnoise(size, corr=0.85, sigma=0.0125, draws=1):
    """Gaussian correlated noise - use for RF if Gauss filter applied."""
    # from BayHunter SynthObs
    idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                          (size, size))
    rmatrix = corr**(idx**2)

    Ce = sigma**2 * rmatrix
    data_noise = rstate.multivariate_normal(np.zeros(size), Ce, draws)

    return np.concatenate(data_noise)


def compute_spectrum(y, Fs):
    """
    Return (normed) single-sided amplitude spectrum of y(t).
    y: receiver function amplitudes
    Fs: Frequency sampling (df) in Hz.
    """
    y = y - np.mean(y)

    n = y.size  # length of the signal
    n_half = int(n/2.)
    k = np.arange(n)
    T = n/Fs
    frq = k/T
    frq = frq[:n_half]  # frequency range

    Y = np.fft.fft(y)/n  # fft computing and normalization
    Y = Y[:n_half]
    Y = abs(Y)
    Y = Y/Y.max()

    return frq, Y


def gauss_fct(a, x):
    """Return Gaussian curve."""
    return np.exp(-(x*2*np.pi)**2 / (4*a**2))


def _min_fct(a, x, y):
    """Minimizing function for Gaussian bell and to be regressed x and y."""
    return gauss_fct(a, x) - y


def _spec_resample(frq, Y):
    """Computes a 2-D histogram of frequencies and spectral energy, and
    resamples the large amount of data to 120 bins.
    """
    bins = 120
    limit = 3  # minimum occurrences per bin
    y_values = np.zeros((bins)) * np.nan  # Y

    hist, xedges, yedges = np.histogram2d(frq, Y, bins=bins)
    xbin = (xedges[:-1] + xedges[1:])/2.
    ybin = (yedges[:-1] + yedges[1:])/2.

    ybin = ybin[::-1]

    histp = hist.T
    histp = histp[::-1]
    for i_y, row in enumerate(histp):
        # going through the rows, starting at top plot (no energy, all frequencies)
        for i_x, occurence in enumerate(row):
            if y_values[i_x] > 0:
                continue
            elif occurence > limit:
                y_values[i_x] = ybin[i_y]

    return xbin, y_values


def plot_rrf_estimate(pars=dict()):
    """ Returns a figure illustrating RF, RF-spectrum, reference Gaussian filter of given
    Gaussfactor [a] and randomly generated noise based on [rrfs] with correlated Gauss 
    factor as displayed in the legend. To estimate a proper rrf for the input RF data,
    choose a rrf-curve that best matches the Gaussian factor used for RF computation.
    As there is still a random factor included (which is reduced by the large amount of 
    drawings), repeat several times to ensure steadiness of your chosen r.
    Please forward all parameters as listed below, otherwise, there might be incompatible
    default values:

    rfx: RF, time in s, default np.linspace(-5, 35, 201)
    rfy: RF, amplitude, default None
    rfa: RF, Gauss factor for label only, default None
    rrfs: rrf to be inspected, default [0.75, 0.85, 0.95]
    a: Gauss factor for reference curve, default 2
    dt: time sampling, default computation from rfx (from rfx default: 0.2)
    draws: number of draws for random noise generation, default 50000

    """
    rfx = pars.get('rfx', np.linspace(-5, 35, 201))
    rfy = pars.get('rfy', None)
    rfa = pars.get('rfa', None)

    rfdt = np.median(np.unique(rfx[1:] - rfx[:-1]))

    rrfs = pars.get('rrfs', [0.75, 0.85, 0.95])
    a = pars.get('a', 2.)  # reference plotting
    dt = pars.get('dt', rfdt)
    df = 1./dt

    fig = plt.figure()

    if rfx is not None and rfy is not None:
        ax_rf = fig.add_subplot(2, 1, 1)
        try:
            label = 'RF, a=%.1f' % rfa
        except Exception:
            label = 'RF'

        ax_rf.plot(rfx, rfy, 'k', lw=1, label=label)
        ax_rf.set_xlabel('Time in s')
        ax_rf.set_ylabel('Amplitude')
        ax_rf.set_xlim(rfx.min(), rfx.max())

        ax_rf.legend(loc=1)

        frq, Y = compute_spectrum(rfy, df)

        ax_p = fig.add_subplot(2, 1, 2)
        ax_p.plot(frq, Y, 'k', lw=1, label='RF-spec', zorder=200)

        print('Time sampling given for RF spectrum: %.3f s' % dt)

    # ----------------------------------------

    else:
        ax_p = fig.add_subplot(1, 1, 1)

    draws = pars.get('draws', 50000)
    sigma = 0.0125  # should not matter
    rrfs = np.array(rrfs)
    rrfs.sort()
    a0 = 1

    Y_all = np.zeros((rrfs.size, int(draws*rfx.size/2.)))

    print('a\trrf')
    for rrf in rrfs:
        rfnoise = _compute_gaussnoise(rfx.size, rrf, sigma, draws=draws)
        frq, Y = compute_spectrum(rfnoise, df)

        # find envelope of spectrum, first resample values
        res_frq, res_Y = _spec_resample(frq, Y)
        res_Y_max = res_Y.max()
        res_Y = res_Y / res_Y.max()

        env_lsq = least_squares(_min_fct, a0, args=(res_frq, res_Y))
        env_a = env_lsq.x[0]
        env_G = gauss_fct(env_a, res_frq)
        label = 'a=%.1f; $r_{RF}$=%.2f' % (env_a, rrf)
        line, = ax_p.plot(res_frq, env_G, lw=1.2, zorder=100, label=label)
        color = line.get_color()

        # ax_p.plot(res_frq, res_Y/res_Y_max, marker='x', zorder=200, lw=1.2,
        #         color=color, label=label)

        ax_p.plot(frq, Y/res_Y_max, lw=0.3, alpha=0.5, color=color)

        print('%.3f\t%.3f' % (env_a, rrf))

    ax_p.set_xlabel('Frequency in Hz')
    ax_p.set_ylabel('Spectral Power')
    ax_p.set_ylim(ymin=0)
    ax_p.set_xlim(frq.min(), frq.max())

    # reference curve for Gaussian 'a' 
    G = gauss_fct(a, res_frq)
    ax_p.plot(res_frq, G, label='a=%.1f' % a, color='k', ls='--', zorder= 200)

    # legend sort by a
    handles, labels = ax_p.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # ax.legend(handles, labels)
    ax_p.legend(handles[::-1], labels[::-1], loc=2, bbox_to_anchor=(1,1.1))

    fig.subplots_adjust(hspace=0.4)
    return fig


def rrf_estimate(pars=dict()):
    """ Returns rrf and a pairs. See explanation for plot_rrf_estimate

    rfx: RF, time in s, default np.linspace(-5, 35, 201)
    rrfs: rrf to be inspected, default [0.75, 0.85, 0.95]
    dt: time sampling, default computation from rfx (from rfx default: 0.2)
    draws: number of draws for random noise generation, default 50000
    """
    rfx = pars.get('rfx', np.linspace(-5, 35, 201))
    rfdt = np.median(np.unique(rfx[1:] - rfx[:-1]))

    rrfs = pars.get('rrfs', [0.75, 0.85, 0.95])
    dt = pars.get('dt', rfdt)
    df = 1./dt

    draws = pars.get('draws', 50000)
    sigma = 0.0125  # should not matter
    rrfs = np.array(rrfs)
    rrfs.sort()
    a0 = 1

    Y_all = np.zeros((rrfs.size, int(draws*rfx.size/2.)))
    a_est = []
    for rrf in rrfs:
        rfnoise = _compute_gaussnoise(rfx.size, rrf, sigma, draws=draws)
        frq, Y = compute_spectrum(rfnoise, df)

        # find envelope of spectrum, first resample values
        res_frq, res_Y = _spec_resample(frq, Y)
        res_Y_max = res_Y.max()
        res_Y = res_Y / res_Y.max()

        env_lsq = least_squares(_min_fct, a0, args=(res_frq, res_Y))
        env_a = env_lsq.x[0]

        a_est.append(env_a)
        print('%.3f\t%.3f' % (env_a, rrf))

    return rrfs, a_est

def calculate_layer_boundaries(z_disc, z_vnoi_pre=None, b=None):
    """
    Convert disc elevations (z_disc) to layer boundaries (z_vnoi) with:
    z_disc[i] = (z_vnoi[i] + z_vnoi[i+1]) / 2

    Parameters:
        z_disc (np.array): Strictly increasing array of disc elevations (>0).
        z_vnoi_pre (np.array, optional): Predefined initial z_vnoi values. 
                                        If provided, the last element of z_vnoi_pre 
                                        becomes the first element of z_vnoi.
        b (float, optional): Free parameter for first boundary (only used if z_vnoi_pre=None).
                            Must satisfy 0 < b < z_disc[0] if provided.
    
    Returns:
        np.array: z_vnoi array where all elements are positive and satisfy the averaging relationship.
    
    Raises:
        ValueError: If input constraints are violated.
    """
    # Input validation for z_disc
    if not (np.all(np.diff(z_disc) > 0) and np.all(z_disc > 0)):
        raise ValueError("z_disc must be strictly increasing and positive")
    
    n_total = len(z_disc) + 1
    z_vnoi = np.zeros(n_total)

    if z_vnoi_pre is not None:
        # ===== 模式1：从 z_vnoi_pre 接续 =====
        if len(z_vnoi_pre) == 0:
            raise ValueError("z_vnoi_pre cannot be empty")
        if not np.all(z_vnoi_pre > 0):
            raise ValueError("All elements in z_vnoi_pre must be positive")
        
        # 将 z_vnoi_pre 的最后一个元素作为 z_vnoi 的第一个元素
        z_vnoi[0] = z_vnoi_pre[-1]
        
        # 计算剩余边界
        for i in range(n_total - 1):
            if i < len(z_disc):
                z_vnoi[i + 1] = 2 * z_disc[i] - z_vnoi[i]
                if z_vnoi[i + 1] <= 0:
                    raise ValueError(f"Generated z_vnoi[{i + 1}] = {z_vnoi[i + 1]} ≤ 0. Adjust z_vnoi_pre or z_disc.")
    else:
        # ===== 模式2：自由参数 b =====
        if b is None:
            b = random.uniform(0.01 * z_disc[0], 0.99 * z_disc[0])  # 避免边界值
        elif not (0 < b < z_disc[0]):
            raise ValueError(f"b must satisfy: 0 < b < {z_disc[0]}")
        
        z_vnoi[0] = b
        for i in range(n_total - 1):
            z_vnoi[i + 1] = 2 * z_disc[i] - z_vnoi[i]
            if z_vnoi[i + 1] <= 0:
                raise ValueError(f"Generated z_vnoi[{i + 1}] = {z_vnoi[i + 1]} ≤ 0. Adjust b or z_disc.")
    
    return z_vnoi

def stack_rf(sum_RRF, sum_TRF):
    '''
    RRF = input sum Stream of rf radial component
    TRF = input sum Stream of rf transverse component
    '''
    
    # Determine minimum length across all RFs
    n_traces = 73
    rf_end = min(len(rf.data) for rf in sum_RRF)
    
    # Initialize arrays
    st_RRF = np.zeros((n_traces, rf_end))
    st_TRF = np.zeros((n_traces, rf_end))
    counts = np.zeros(n_traces)
    
    # Stack RFs into appropriate bins
    for rrf, trf in zip(sum_RRF, sum_TRF):
        baz = rrf.stats.sac['baz']
        
        # Assign to 5-degree bin (0-360° → 0-72 bins)
        trace = int(round(baz / 5)) % n_traces
        
        st_RRF[trace, :] += rrf.data[:rf_end]
        st_TRF[trace, :] += trf.data[:rf_end]
        counts[trace] += 1
    # Compute averages (avoid division by zero)
    valid_traces = counts > 0
    st_RRF[valid_traces, :] /= counts[valid_traces, None]
    st_TRF[valid_traces, :] /= counts[valid_traces, None]
            
    # Combine 0° and 360° (bin 0 and bin 72)
    st_RRF[0, :] = st_RRF[-1, :] = (st_RRF[0, :] + st_RRF[-1, :]) / max(counts[0] + counts[-1], 1)
    st_TRF[0, :] = st_TRF[-1, :] = (st_TRF[0, :] + st_TRF[-1, :]) / max(counts[0] + counts[-1], 1)
    
    # Combine RRF and TRF
    stacked_data = np.concatenate((st_RRF, st_TRF), axis=1)

    # Create 0/1 mask
    mask = np.where(counts > 0, 1, 0)

    return stacked_data, mask, counts

def read_paired_q_t_streams(sac_dir, target_delta=0.1, end_time=6.0, max_amplitude=1.0):
    """
    读取配对的Q和T分量SAC文件，降采样并截断到指定时间，剔除振幅过大的波形对。

    参数:
        sac_dir (str): SAC文件目录路径。
        target_delta (float): 目标采样间隔（秒），必须为原始delta的整数倍。
        end_time (float): 截断时间（秒），如6.0表示只返回6秒前的数据。
        max_amplitude (float): 允许的最大振幅绝对值，超过此值的波形对将被剔除。

    返回:
        tuple: (q_stream, t_stream, x_axis)
            - q_stream: 降采样并截断后的Q分量Stream
            - t_stream: 降采样并截断后的T分量Stream
            - x_axis: 截断后的时间轴（从b开始到end_time）
    """
    # 读取原始数据并降采样
    q_stream, t_stream, x_axis_full = read_paired_q_t_streams_raw(sac_dir, target_delta)
    
    # 获取第一个Trace的起始时间b
    b = q_stream[0].stats.sac.get('b', -1.0)
    
    # 计算截断的样本数
    npts_truncate = int((end_time - b) / target_delta) + 1
    
    # 截断时间轴
    x_axis = x_axis_full[:npts_truncate]
    
    # 创建新的Stream对象用于存储有效数据
    valid_q_stream = Stream()
    valid_t_stream = Stream()
    
    # 检查每个波形对
    for q_trace, t_trace in zip(q_stream, t_stream):
        # 检查Q和T分量是否都未超过最大振幅
        q_max = np.max(np.abs(q_trace.data[:npts_truncate]))
        t_max = np.max(np.abs(t_trace.data[:npts_truncate]))
        
        if q_max <= max_amplitude and t_max <= max_amplitude:
            # 创建新的Trace对象，避免修改原始数据
            new_q_trace = q_trace.copy()
            new_t_trace = t_trace.copy()
            
            # 截断数据
            new_q_trace.data = new_q_trace.data[:npts_truncate]
            new_t_trace.data = new_t_trace.data[:npts_truncate]
            
            # 添加到有效流中
            valid_q_stream.append(new_q_trace)
            valid_t_stream.append(new_t_trace)
    
    return valid_q_stream, valid_t_stream, x_axis

def read_paired_q_t_streams_raw(sac_dir, target_delta):
    """（原函数逻辑，用于内部调用）"""
    all_files = os.listdir(sac_dir)
    prefixes = set()
    for f in all_files:
        if f.endswith(".Q.SAC") or f.endswith(".T.SAC"):
            prefix = f.rsplit(".", 2)[0]
            prefixes.add(prefix)
    
    q_stream = Stream()
    t_stream = Stream()
    
    for prefix in sorted(prefixes):
        q_file = os.path.join(sac_dir, f"{prefix}.Q.SAC")
        t_file = os.path.join(sac_dir, f"{prefix}.T.SAC")
        
        if os.path.exists(q_file):
            q_stream += obspy.read(q_file)
        if os.path.exists(t_file):
            t_stream += obspy.read(t_file)
    
    # 降采样
    first_trace = q_stream[0]
    original_delta = first_trace.stats.delta
    decimation_factor = int(round(target_delta / original_delta))
    q_stream.decimate(decimation_factor, no_filter=True)
    t_stream.decimate(decimation_factor, no_filter=True)
    
    # 生成完整时间轴
    b = first_trace.stats.sac.get('b', -1.0)
    npts = len(q_stream[0].data)
    x_axis_full = b + np.arange(npts) * target_delta
    
    return q_stream, t_stream, x_axis_full

def remove_reverberations(RF, dt, water_level=0.05, return_params=False):
    """
    去除接收函数中的沉积层多次反射（共振移除滤波）
    
    参数:
    ----------
    RF : array_like
        原始接收函数时间序列（含多次反射）
    dt : float
        时间采样间隔（秒）
    water_level : float, optional
        水位反卷积的水位值（默认 0.05）
    return_params : bool, optional
        是否返回滤波器参数 r0 和 Delta_t（默认 False）
    
    返回:
    ----------
    RF_remove : ndarray
        去除多次反射后的接收函数
    (r0, Delta_t) : tuple (可选)
        仅当 return_params=True 时返回，多次反射参数
    """
    # --- 1. 计算自相关函数，提取 r0 和 Delta_t ---
    autocorr = correlate(RF, RF, mode='full')
    autocorr = autocorr[len(RF)-1:]  # 取非负延迟部分
    autocorr = autocorr / autocorr[0]  # 归一化
    
    # 寻找第一个波谷（多次反射周期）
    troughs, _ = find_peaks(-autocorr, height=-0.1)  # 忽略极小波谷
    if len(troughs) == 0:
        print("Warning: 未检测到明显多次反射，返回原始接收函数")
        return RF if not return_params else (RF, (0, 0))
    
    Delta_t = troughs[0] * dt  # 多次反射周期
    r0 = -autocorr[troughs[0]]  # 反射系数（取正值）
    
    # --- 2. 构建频率域滤波器 (1 + r0 * exp(-iωΔt)) ---
    n = len(RF)
    freqs = fftfreq(n, dt)
    omega = 2 * np.pi * freqs
    filter_fft = 1 + r0 * np.exp(-1j * omega * Delta_t)
    
    # --- 3. 应用滤波器 ---
    RF_fft = fft(RF)
    RF_remove_fft = RF_fft * filter_fft
    RF_remove = np.real(ifft(RF_remove_fft))
    
    # --- 4. 可选：应用水位限制（防止高频噪声放大）---
#     if water_level is not None:
#         RF_remove = np.where(np.abs(RF_remove) < water_level, 
#                             np.sign(RF_remove) * water_level, 
#                             RF_remove)
    
    return (RF_remove, (r0, Delta_t)) if return_params else RF_remove

def plot_comparison(xrf, yrf, scale=10,
                    leftlabel=True, order=None, title=None):
    """
    Plot sorted traces on two panels, 'Radial and Transverse' or
    'Fast and Slow'. Emphasizing for a specific time window also
    avilible.
    ----------
    profile_type: str
        'original': input Radial- and Transverse-component Stream
        'corrected': corrected Radial- and Transverse-component Stream
        'fastslow': corrected Fast- and Slow-component Stream
        'beforeshift': Fast- and Slow-component Stream before time delay shift
    scale: float
    timemin: float
        start time of time-window
    timemax: float
        end time of time-window
    emphasize: bool
        To high-light signal within a specific time-window
        defined by 'emphamin' and 'emphamax'
    """
    fig, ax = plt.subplots(figsize=(16.0,16.0),dpi=330)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if order!=None:
        plt.figtext(order[0], order[1], order[2],
                    horizontalalignment="center", 
                    verticalalignment="center",
                    wrap=True, fontsize=36)
    
    ax1.set_xticks(np.arange(0, 20, 1))
    ax1.set_xlim(xmin=-1)
    ax1.set_xlabel('Time [s]', fontsize=46)
    ax1.set_ylim(ymin=-10, ymax=370)
    if leftlabel:
        ax1.set_yticks(np.linspace(0,360,13))
        ax1.set_ylabel('Back Azimuth [°]', fontsize=46)
    else:
        ax1.set_yticks([])
    ax1.tick_params(size=7, width=2, labelsize=36)
    
    ax2.set_xticks(np.arange(0, 20, 1))
    ax2.set_xlim(xmin=-1)
    ax2.set_ylim(ymin=-10, ymax=370)
    ax2.set_yticks([])
    ax2.set_xlabel('Time [s]', fontsize=46)
    ax2.tick_params(size=7, width=2, labelsize=36)
    
    # Set boundary
    for key in ['top', 'bottom', 'left', 'right']:
        ax1.spines[key].set_linewidth(3.5)
        ax2.spines[key].set_linewidth(3.5)
    
    ax1.axvline(x=0 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax1.axvline(x=2 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax1.axvline(x=4 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax1.axvline(x=6 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax2.axvline(x=0 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax2.axvline(x=2 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax2.axvline(x=4 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    ax2.axvline(x=6 , color='black' , linestyle='--', alpha=0.4, linewidth=2)
    
    alpha_em = 1

    for i, iyrf in enumerate(yrf[:, :len(xrf)]):
        baz = i * 5
        data_set_1 = iyrf[:len(xrf)] * scale + baz
        data_set_2 = iyrf[len(xrf):] * scale + baz
        
        ax1.plot(xrf, data_set_1, color='k', alpha=0.1)
        ax1.fill_between(xrf, data_set_1,
                         baz, where=(data_set_1>baz), facecolor='b', alpha=alpha_em)
        ax1.fill_between(xrf, data_set_1,
                         baz, where=(data_set_1<baz), facecolor='r', alpha=alpha_em)
        
        ax2.plot(xrf, data_set_2,color='k', alpha=0.1)
        ax2.fill_between(xrf, data_set_2,
                         baz, where=(data_set_2>baz), facecolor='b', alpha=alpha_em)
        ax2.fill_between(xrf, data_set_2,
                         baz, where=(data_set_2<baz), facecolor='r', alpha=alpha_em)
    plt.tight_layout()
    plt.show()
    
    
    return fig, 