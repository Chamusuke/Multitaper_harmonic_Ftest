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
    Slepian シーケンス (DPSS: Discrete Prolate Spheroidal Sequences) の作成
    Generate Slepian sequences (DPSS: Discrete Prolate Spheroidal Sequences)

    Parameters
    ----------
    npts : int
        テーパーのポイント数(データ長)
        Number of points in each taper (data length)

    nw : float
        タイムバンド幅積 (time-bandwidth product)
        Time-bandwidth product (NW)

    k : int, optional
        デフォルト値 -> 2*nw - 1
        Default -> 2*nw - 1
        もし 2*nw が整数の場合、固有値が 0.90 以上の DPSS を採用
        If 2*nw is integer, adopt DPSS with eigenvalues >= 0.90

    Returns
    -------
    dpss : ndarray [k, npts]
        K 個のテーパー群
        Array of K tapers (DPSS sequences)

    eigenvalues : ndarray [k]
        K 個の固有値(帯域幅内のエネルギー集中度)
        Array of K eigenvalues (measure of spectral concentration within bandwidth)
    """

    if k is None:
        k = int(2*nw-1)
           
    k_DPSS, eigenvalues = windows.dpss(
        npts, nw, Kmax=k, sym=False, norm=2, return_ratios=True
    )

    # 固有値フィルタリング
    if k >= 2 * nw:
        valid_indices = np.where(eigenvalues >= 0.90)[0]
        k_DPSS = k_DPSS[valid_indices]
        eigenvalues = eigenvalues[valid_indices]
        k = len(valid_indices)
        print(f"A unique taper of eigenvalues (>=0.90) was selected. K={k}")

    # -----------------------------------------------------
    # 正規化(念のため再確認)
    # -----------------------------------------------------
    vnorm = np.sqrt(np.sum(k_DPSS**2, axis=1))  # 各テーパーのL2ノルム
    k_DPSS = k_DPSS / vnorm[:, None]            # 正規化して単位エネルギーに

    # -----------------------------------------------------
    # 符号統一(positive-standard)
    # -----------------------------------------------------
    nx = npts % 2
    if nx == 1:
        lh = (npts + 1) // 2
    else:
        lh = npts // 2

    for i in range(k):
        if k_DPSS[i, lh] < 0.0:
            k_DPSS[i, :] = -k_DPSS[i, :]

    return k_DPSS, eigenvalues, k

def eigen_psd(data, k_DPSS, fs, nfft):
    """
    各テーパーごとの固有周波数スペクトルおよび固有パワースペクトルを計算
    Compute eigenfrequency spectra and eigen power spectra for each taper

    Parameters
    ----------
    data : ndarray
        入力時系列データ
        Input time series data

    k_DPSS : ndarray
        DPSS(Slepianシーケンス)ターパー群
        Set of DPSS (Slepian sequence) tapers

    fs : float
        サンプリング周波数 [Hz]
        Sampling frequency [Hz]

    nfft : int
        FFT長(スペクトル分解能を決定)
        FFT length (determines spectral resolution)

    Returns
    -------
    eigenspectra : ndarray
        各ターパーごとの周波数スペクトル(複素スペクトル)
        Frequency spectra (complex) for each taper

    eigen_psd : ndarray
        各ターパーごとのパワースペクトル密度(PSD)
        Power spectral density (PSD) for each taper
    """

    tapered_data = k_DPSS * data[np.newaxis, :]  # shape: (K, N)
    Jk = np.fft.fft(tapered_data, n=nfft, axis=1) * np.sqrt(1/fs) 
    Sn_k = np.abs(Jk)**2
    return Jk, Sn_k


class MultiTaper_Periodogram:
    """
    DPSS を用いたマルチターパー法による PSD 推定を行うクラス
    Class for PSD estimation using DPSS and the Multitaper Method
    """
    def __init__(
        self,
        NW: float = 4.0, 
        K: int = None,
        nfft: int = None,
        detrend: str = 'constant',
        p_level: float = 0.05
    ):
        """
        Parameters
        ----------
        fs : float
            サンプリング周波数 [Hz]
            Sampling frequency [Hz]

        NW : float, optional
            タイムバンド幅積 (time-bandwidth product)
            Time-bandwidth product (NW)

        K : int, optional
            DPSS テーパー数
            Number of DPSS tapers

        nfft : int, optional
            FFT のサンプル数
            Number of FFT points
            データ長以下なら信号の一部分で解析、以上ならゼロ埋めによりスペクトルを滑らかに
            If less than data length, analyze part of the signal; if greater, apply zero-padding for smoother spectrum

        detrend : {'linear', 'constant'}, None optional
            除去するトレンドの種類
            Type of trend to remove
                'linear': 一次直線を最小二乗フィットして引き算
                          Fit and subtract a linear trend (default)
                'constant': default データの平均値 (DC成分) を引き算
                            Subtract mean value (DC component)

        p_level : float, optional
            有意水準 (default 0.05)
            Significance level (default 0.05)
        """
        self.NW = NW
        self.K = K
        self.nfft = nfft
        self.detrend = detrend



    def MT_Spec(self, data: np.ndarray, fs: float):
        """
        与えられた時系列データに対して、
        マルチターパー (DPSS) 法を用いてパワースペクトル密度 (PSD) を推定する。
        Estimate the Power Spectral Density (PSD) of a given time series
        using the Multitaper method with DPSS tapers.

        Parameters
        ----------
        data : ndarray
            1次元の時系列データ
            One-dimensional time series data

        fs : float
            サンプリング周波数 [Hz]
            Sampling frequency [Hz]

        Returns
        -------
        f : ndarray
            周波数軸 [Hz]
            Frequency axis [Hz]

        Smt : ndarray
            推定されたパワースペクトル (長さは 'f' と同じ)
            Estimated power spectral density (same length as 'f')
        """

        self.data = np.asarray(data)
        self.fs = fs
        self.N = len(data)

        # DPSS テーパーの生成
        self.k_DPSS, self.eigenvalues, self.K = dpss(self.N, self.NW, self.K)

        # detrend
        self.data = detrend(data, self.detrend)
        self.xvar = np.var(self.data)

        # MT法によるスペクトル推定
        if self.nfft is None:
            self.nfft = len(data)
        self.Jk, self.Sn_k = eigen_psd(self.data, self.k_DPSS, self.fs, self.nfft)  # (K, nfft)
        self.Sn = np.mean(self.Sn_k, axis=0)

        # 周波数軸とスペクトル化の場合分け
        if np.iscomplexobj(self.data):
            # 複素信号 → 両側スペクトル
            self.f = np.fft.fftfreq(self.nfft, d=1/self.fs)
            nfreq = len(self.f)
            self.Smt_k = self.Sn_k[:, :nfreq]
            self.Smt = np.mean(self.Smt_k, axis=0)
        else:
            # 実信号 → 片側スペクトル
            self.f = np.fft.rfftfreq(self.nfft, d=1/self.fs)
            nfreq = len(self.f)
            self.Smt_k = self.Sn_k[:, :nfreq]
            self.Smt = np.mean(self.Smt_k, axis=0)
            # DCとNyquist以外を2倍補正
            self.Smt_k[:, 1:-1] *= 2
            self.Smt[1:-1] *= 2

        # Parseval スケール(片側補正後に計算)
        df = self.fs / self.nfft
        self.sscal = self.xvar / (np.sum(self.Smt) * df)
        self.Smt   = self.Smt * self.sscal
        self.Smt_k = self.Smt_k * self.sscal
        return self.f, self.Smt


    def Harmonic_Ftest(self, p_level: float = 0.05):
        """
        Harmonic F-test for detecting significant sinusoidal components
        using Multitaper spectral estimates.

        Parameters
        ----------
        p_level : float
            Significance level (default 0.05)
        
        Returns
        -------
        re_psd : ndarray [3, nfft]
            self.re_psd[0, :] -> 背景PSD (Background PSD)
            self.re_psd[1, :] -> 線成分PSD (Line component PSD)
            self.re_psd[2, :] -> 再構成PSD (Reconstructed multitaper PSD)

        Notes
        -----
        F_stat と F_crit は Return には含まれず、クラス属性として利用可能。
        F_stat and F_crit are not included in the return values,
        but can be accessed directly as class attributes:
            <Object>.F_stat
            <Object>.F_crit
        """


        npts  = np.shape(self.k_DPSS)[1]    # DPSSターパーの長さ
        kspec = np.shape(self.k_DPSS)[0]    # DPSSターパーの本数
        # W_bin の計算例
        df = self.fs / self.nfft   # 周波数分解能
        W_hz = (self.NW * self.fs) / npts
        W_bin = int(W_hz / df)

        dof1 = 2
        dof2 = 2 * (kspec - 1)

        # 各テーパーにおける H_k(0) の算出
        H_k0 = (1/self.fs) * np.sum(self.k_DPSS, axis=1)
        H_k0[1::2] = 0  # 奇関数の和は`0なので明記

        # 回帰係数 C の算出
        H_k0_2sum = np.sum(H_k0**2)
        Jk_Hk0 = np.sum(self.Jk * H_k0[:, np.newaxis], axis=0)
        C = np.sqrt(1/self.fs) * Jk_Hk0 / H_k0_2sum

        # F統計量の計算
        Fup = float(kspec - 1) * H_k0_2sum * np.abs(C)**2
        Jk_hat_1 = (C * H_k0[:, np.newaxis]) / np.sqrt(1/self.fs)
        Fdw = (1/self.fs) * np.sum(np.abs(self.Jk - Jk_hat_1)**2, axis=0)
        F = Fup / Fdw

        # p値の計算
        p = stats.f.cdf(F, dof1, dof2)

        self.F_stat = np.zeros((2, self.nfft), dtype=float)
        self.F_stat[0, :] = F
        self.F_stat[1, :] = p

        self.F_crit = stats.f.ppf(1 - p_level, dof1, dof2)
        self.F_999 = stats.f.ppf(0.999, dof1, dof2)
        self.F_99 = stats.f.ppf(0.99 - p_level, dof1, dof2)

        # 有意な周波数を取得
        p_masked = np.where(p > (1 - p_level), p, 0.0)
        local_maxima, _ = signal.find_peaks(p_masked, plateau_size=1)
        nl = len(local_maxima)

        # スペクトル再構成
        self.re_psd = np.zeros((3, self.nfft), dtype=float)

        if nl == 0:
            # 有意な線スペクトルがない場合
            self.re_psd[0, :] = self.Sn
            self.re_psd[1, :] = np.zeros(self.nfft, dtype=float)
            self.re_psd[2, :] = self.Sn
            self.k_psd_back = np.copy(self.Sn_k)
        else:
            # 有意な線スペクトルのみ残す
            C_test = np.zeros_like(C)
            C_test[local_maxima] = C[local_maxima]

            H_k = (1/self.fs) * np.fft.fft(self.k_DPSS, n=self.nfft, axis=1)
            back_Jk = np.copy(self.Jk)
            Jk_line = np.zeros((kspec, self.nfft), dtype=complex)

            for i in local_maxima:
                # ピーク近傍の周波数帯を定義
                f_start = max(i - W_bin, 0)
                f_end   = min(i + W_bin + 1, self.nfft)
                f_band  = np.arange(f_start, f_end)

                # H_k(f - f1) をバンド内で評価
                jj = (f_band - i) % self.nfft
                delta = (C_test[i] / np.sqrt(1/self.fs)) * H_k[:, jj]
                back_Jk[:, f_band] -= delta
                 # 帯域で引いた総エネルギー(ターパーごと)
                E_removed = np.sum(delta, axis=1)                      # shape: (K,)
                # ガウス窓(ピーク中心)。総和=1 に正規化して分配
                x = np.arange(len(f_band)) - (len(f_band)//2)
                sigma = W_bin / 1e6                                    # 幅は目的に応じて調整
                W = np.exp(-(x**2) / (2*sigma**2))                     # shape: (|f_band|,)
                W /= np.sum(W)                                         # 総和=1
                # 引いた総エネルギーをガウス窓に従って線成分へ再分配(厳密保存)
                Jk_line[:, f_band] += E_removed[:, None] * W[None, :]


            k_psd_back = (np.abs(back_Jk))**2
            psd_back = np.mean(k_psd_back, axis=0)

            S_line =np.mean(np.abs(Jk_line)**2, axis=0)

            Jk_re = back_Jk + Jk_line
            re_mt_psd =np.mean(np.abs(Jk_re)**2, axis=0)

            self.re_psd[0, :] = psd_back
            self.re_psd[1, :] = S_line
            self.re_psd[2, :] = re_mt_psd
            self.re_psd = self.re_psd *self.sscal


        # 切り出し
        nfreq = len(self.f)  # rfftfreqなら nfft//2+1
        self.re_psd = self.re_psd[:,:nfreq]
        self.F_stat  = self.F_stat[:, :nfreq]
        
        if not np.iscomplexobj(self.data) and nfreq > 2:
            self.re_psd[:,1:-1] *= 2 

        return self.re_psd


