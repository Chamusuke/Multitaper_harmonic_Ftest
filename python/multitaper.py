#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
from scipy import signal,stats
from scipy.signal import periodogram, windows


def detrend(data, option = 'constant'):
    if option == 'constant':
        output = (data - np.mean(data))
    elif option == 'linear':
        n = len(data)  # データの長さ
        x = np.arange(n)  # 0, 1, 2, ..., n-1 のインデックスを作成
        a, b = np.polyfit(x, data, 1)
        output = data - (a * x + b)
    else:
        output = data
    return output


def dpss(npts, nw, k=None):
    """
    Slepian シーケンス(DPSS)の作成

    Parameters
    ----------
    npts : int
        テーパーのポイント数 (データ長)
    nw : float, optional
        タイムバンド幅積 (time-bandwidth product)        
    k : int
        defalt -> 2*nw-1
        2*nw異能の場合,固有値0.8以上のDPSSを採用する
    
    **Return*
    dpss : ndarray [k, npts]
        K個のテーパー群
    eigenvalues : adarray [k]
        K個の固有値 帯域幅のエネルギー集中尺度
    
    """
    if k is None:
        k = int(2*nw-1)
           
    k_DPSS, eigenvalues = windows.dpss(npts, nw, Kmax=k, sym=False, norm=2, return_ratios=True)

    if k >= 2 * nw:
        valid_indices = np.where(eigenvalues >= 0.8)[0]
        k_DPSS = k_DPSS[valid_indices]
        eigenvalues = eigenvalues[valid_indices]
        k = len(valid_indices)
        print(f"A unique taper of 0.8 or more was selected. K={k}")

    return k_DPSS, eigenvalues, k

def eigenspec(data, k_DPSS, nfft):
    """
    各テーパーごとの固有周波数スペクトル及び固有パワースペクトル密度
    (Eigen Spectrum and PSD of each DPSS)
    Jk : 固有周波数スペクトル (Eigen Spectrum)
    Sk : 固有パワースペクトル密度 (Eigen PSD)
    """
    tapered_data = k_DPSS * data[np.newaxis, :]  # shape: (K, N)
    Jk = np.fft.fft(tapered_data, n=nfft, axis=1)  # shape: (K, nfft)
    Sk = (np.abs(Jk))**2
    return Jk, Sk


class MultiTaper_Periodogram:
    """
    DPSS を用いたマルチターパー法での PSD 推定を行うクラス
    this class is estimation of PSD using DPSS and Multitaper Method
    """
    def __init__(
        self,
        NW: float = 4.0, 
        K: int = None,
        nfft: int = None,
        detrend: str = 'constant',
    ):
        """
        Parameters
        ----------
        fs : float
            サンプリング周波数 (Sampling Frequency) [Hz]
        NW : float, optional
            タイムバンド幅積 (time-bandwidth product)
        K : int, optional
            DPSS テーパー数  (DPSS Tapers)
        nfft : int, optional
            FFT のサンプル数 (Point od FFT))
            データ長 以下なら信号の一部分で解析 以上なら0うめにより滑らかに
        scaling : {'density', 'spectrum'}, optional
            - 'density' : パワースペクトル密度 (PSD) [V^2/Hz]
            - 'spectrum': パワースペクトル [V^2]
        return_onesided : bool, optional
            True なら片側スペクトルを返す   positive frequency
        detrend :  { 'linear', 'constant' }, None optional
            除去するトレンドの種類
                'linear' (デフォルト): 一次の直線を最小二乗フィットして、それを引き算
            Def 'constant': データの平均値(0次多項式 DC成分)を引き算
        p_level: 
            有意水準 default 0.05
        """
        self.NW = NW
        self.K = K
        self.nfft = nfft
        self.detrend = detrend
        self.eigenvalue = None
        self.k_DPSS = None
        self.f = None
        self.fs = None
        self.mt_psd = None
        self.re_psd = None
        self.re_psd_sline = None
        self.re_psd_sback = None



    def MT_Spec(self, data: np.ndarray, fs:float):
        """
        与えられた時系列データ x に対して、
        マルチターパー (DPSS) 法で PSD を推定する。
        We can estimate PSD using Multitaper and DPSS

        Parameters
        ----------
        x : ndarray
            1次元の時系列データ
            Time Series data

        Returns
        -------
        f : ndarray
            周波数軸 (Frequency) [Hz]
        Pxx : ndarray
            推定されたパワースペクトル (same data lenge 'f')
        """
        self.data = np.asarray(data)
        self.fs = fs
        
        npts = len(data)
        # DPSS テーパーの生成 (shape: (K, N))
        self.k_DPSS, self.eigenvalues, self.K = dpss(npts, self.NW, self.K)

        # detrend
        self.data = detrend(data,self.detrend)

        # MT法によるスペクトル推定
        if self.nfft is None:
            self.nfft = len(data)

        self.Jk, k_psd = eigenspec(self.data, self.k_DPSS, self.nfft) # (K, nfft)
        mt_psd = np.mean(k_psd, axis=0)
        f = np.fft.fftfreq(self.nfft, d=1/self.fs)

        # 片側スペクトル(dc,ナイキストは除いて補正)
        self.k_psd = k_psd[:,:self.nfft // 2]
        self.mt_psd = mt_psd[:self.nfft // 2]

        self.k_psd[:, 1:-1] *= 2  
        self.mt_psd[1:-1] *= 2    

        self.f = f[:self.nfft // 2]

        return None


    def Harmonic_Ftest(self, p_level):
        # complex_asd_list < yk
        # Vn < DPSS

        npts  = np.shape(self.k_DPSS)[1]    # DPSSターパーの長さ (サンプル数)
        kspec = np.shape(self.k_DPSS)[0]    # DPSSターパーの本数

        C    = np.zeros(self.nfft, dtype=complex)
        F     = np.zeros(self.nfft)
        p     = np.zeros(self.nfft)

        dof1 = 2
        dof2 = 2 * (kspec - 1)

        #各テーパーにおけるH_k(0)の算出
        H_k0 = np.sum(self.k_DPSS, axis=1)  #shape: (k,)
        H_k0[1::2] = 0  # 奇関数の和は0にする

         # 各周波数における回帰係数 C/sqrt(dt) の算出
        H_k0_2sum = np.sum(H_k0**2)

        Jk_Hk0 = np.sum(self.Jk * H_k0[:, np.newaxis], axis=0)
        C = np.sqrt(1 / self.fs) * Jk_Hk0 / H_k0_2sum


        # F統計量  
        # 分子（Fup）の計算
        Fup = float(kspec - 1) * np.abs(C) ** 2 * H_k0_2sum  # shape: (nfft,)

        # 残差の計算（Fdw）
        Fdw = (1/self.fs)*np.sum(np.abs(self.Jk - (C * H_k0[:, np.newaxis]) / np.sqrt(1/self.fs))**2, axis=0)  # shape: (nfft,)

        # F値の計算
        F = Fup / Fdw  # shape: (nfft,)
        F = F[:self.nfft // 2]
        # p値の計算
        p = stats.f.cdf(F, dof1, dof2)  # shape: (nfft,)
        p = p[:self.nfft // 2]

        self.F = F[:,np.newaxis]
        # self.p = p[:,np.newaxis]
        self.F_crit = stats.f.ppf(1 - p_level, dof1, dof2)

        # スペクトル再構成
        sline = np.zeros( (self.nfft,1), dtype=float)
        JK =  np.zeros( (kspec, self.nfft), dtype=complex)


        # 有意な周波数を取得
        p[p < (1-p_level)] = 0

        local_maxima = signal.argrelextrema(p, np.greater)[0]  # 局所最大値のインデックスを取得

        # 局所最大値でないものをゼロにする
        filtered_p = np.zeros_like(p)
        filtered_p[local_maxima] = p[local_maxima]

        # ピークの数をカウント
        nl = len(local_maxima)

        # if (nl == 0): 
            # return seJk, sline

        return None
