#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
from scipy import signal,stats
from scipy.signal import periodogram, windows


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
        scaling: str = 'density',
        return_onesided: bool = True,
        detrend: bool = 'constant',
        debug: int = 1,
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
        detrend :  { 'linear', 'constant' }, False optional
            除去するトレンドの種類
                'linear' (デフォルト): 一次の直線を最小二乗フィットして、それを引き算
            Def 'constant': データの平均値(0次多項式 DC成分)を引き算
                'linear' を使うと、より長期的な勾配成分（スロープ）を除去
        p_level: 
            有意水準 default 0.05
        """
        self.NW = NW
        self.K = K
        self.nfft = nfft
        self.scaling = scaling
        self.return_onesided = return_onesided
        self.detrend = detrend
        self.eigenvalue = None
        self.k_DPSS = None
        self.k_DPSS_Pxx = []
        self.f = None
        self.mt_psd = None
        self.re_psd = None
        self.re_psd_sline = None
        self.re_psd_sback = None

    def Z_nomalization(data):
        output = (data - np.mean(data)) / np.std(data)
        return output
    

    def dpss(self, npts, nw, k=None):
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
        
        **Return*
        dpss : ndarray [k, npts]
            K個のテーパー群
        eigenvalues : adarray [k]
            K個の固有値 帯域幅のエネルギー集中尺度
     
        """
        if k is None:
            k = int(2*nw-1)
            self.K = k

        self.k_DPSS, self.eigenvalues = windows.dpss(npts, nw, Kmax=k, sym=False, norm=2, return_ratios=True)

        # nx = npts%2
        # if (nx==1):
        #     lh = int((npts+1)/2)
        # else:
        #     lh = int(npts/2)

        # for k_i in range(self.K):
        #     if (self.k_DPSS[k_i,lh] < 0.0):
        #         self.k_DPSS[k_i,:] = -self.k_DPSS[k_i,:]

        return None
    


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
        
        N = len(data)
        # DPSS テーパーの生成 (shape: (K, N))
        self.dpss(N, self.NW, self.K)

        if self.nfft is None:
            self.nfft = len(data)

        psd_list = []
        # self.k_DPSS.shape = (K, N)
        for k_i in range(self.K):
            # テーパーを用いてperiodogram を計算
            f, Pxx = periodogram(
                data,
                fs=self.fs,
                window=self.k_DPSS[k_i, :], 
                nfft=self.nfft,
                detrend=self.detrend,
                return_onesided=False,
                scaling=self.scaling
            )
                # 片側スペクトルへ変換
            Pxx_onesided = Pxx[:self.nfft // 2 - 1]  # 正の周波数成分のみ取得
            Pxx_onesided[1:-1] *= 2  # エネルギー保存のため2倍
            psd_list.append(Pxx_onesided)

        self.k_DPSS_Pxx = np.stack(psd_list, axis=0)  # shape: (K, Nfreq)
        self.mt_psd = np.mean(self.k_DPSS_Pxx, axis=0)  # 周波数方向の平均
        self.f = f[:self.nfft // 2 - 1]

        return None


    def Harmonic_Ftest(self, p_level):
        # complex_asd_list < yk
        # Vn < DPSS


        npts  = np.shape(self.k_DPSS)[1]    # DPSSターパーの長さ (サンプル数)
        kspec = np.shape(self.k_DPSS)[0]    # DPSSターパーの本数

        Jk = [] #(K, nfft)
        # self.k_DPSS.shape = (K, N)
        # テーパ群の適用
        tapered_data = self.k_DPSS * self.data[np.newaxis, :]  # shape: (K, N)
        # FFTをベクトル化**
        Jk = np.fft.fft(tapered_data, n=self.nfft, axis=1)  # shape: (K, nfft)

        C    = np.zeros(self.nfft, dtype=complex)
        F     = np.zeros(self.nfft)
        p     = np.zeros(self.nfft)

        dof1 = 2
        dof2 = 2 * (kspec - 1)

        #各テーパーにおけるH_k(0)の算出
        H_k0 = np.sum(self.k_DPSS, axis=1)  # 各行の合計を計算
        H_k0[1::2] = 0  # 奇関数の和は0にする

         # 各周波数における回帰係数 C/sqrt(dt) の算出
        H_k0_2sum = np.sum(H_k0**2)

        Jk_Hk0 = np.sum(Jk * H_k0[:, np.newaxis], axis=0)
        C = np.sqrt(1 / self.fs) * Jk_Hk0 / H_k0_2sum


        # F統計量  
        # 分子（Fup）の計算
        Fup = float(kspec - 1) * np.abs(C) ** 2 * H_k0_2sum  # shape: (nfft,)

        # 残差の計算（Fdw）
        Fdw = (1/self.fs)*np.sum(np.abs(Jk - (C * H_k0[:, np.newaxis]) / np.sqrt(1/self.fs))**2, axis=0)  # shape: (nfft,)

        # F値の計算
        F = Fup / Fdw  # shape: (nfft,)
        F = F[:self.nfft // 2 - 1]
        # p値の計算
        p = stats.f.cdf(F, dof1, dof2)  # shape: (nfft,)
        p = p[:self.nfft // 2 - 1]

        self.F = F[:,np.newaxis]
        self.p = p[:,np.newaxis]
        self.F_crit = stats.f.ppf(1 - p_level, dof1, dof2)

        return None
