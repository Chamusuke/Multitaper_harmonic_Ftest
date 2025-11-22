#-----------------------------------------------------
# Import main libraries and modules
#-----------------------------------------------------

import numpy as np
from scipy import signal,stats
from scipy.signal import periodogram, windows


def get_data(fname):
    """
    Utility function to download the data from the Zenodo repository
    with the direct URL path (fixed). 
    
    **Parameters**
    
    fname : char
        filename of the data to download
    
    **Returns**
    
    data : ndarray
        numpy array with the downloaded data
        In case of error, data = 0 is returned

    |

    """
    
    if (fname.find("v22")>-1):
        url = 'https://zenodo.org/record/6025794/files/v22_174_series.dat?download=1'
    elif (fname.find("hhe.dat")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_vmm_hhe.dat?download=1'
    elif (fname.find("sgc_vmm.dat")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_vmm.dat?download=1'
    elif (fname.find("sgc_surf")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_surf.dat?download=1'
    elif (fname.find("sgc_mesetas")>-1):
        url = 'https://zenodo.org/record/6025794/files/sgc_mesetas.dat?download=1'
    elif (fname.find("PASC")>-1):
        url = 'https://zenodo.org/record/6025794/files/PASC.dat?download=1'
    elif (fname.find("_src")>-1):
        url = 'https://zenodo.org/record/6025794/files/mesetas_src.dat?download=1'
    elif (fname.find("crisanto")>-1):
        url = 'https://zenodo.org/record/6025794/files/crisanto_mesetas.dat?download=1'
    elif (fname.find("akima")>-1):
        url = 'https://zenodo.org/record/6025794/files/asc_akima.dat?download=1'
    elif (fname.find("ADO")>-1):
        url = 'https://zenodo.org/record/6025794/files/ADO.dat?download=1'
    else:
        data = -1
        
    data = np.loadtxt(url)
    
    return data

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
        2*nw異能の場合,固有値0.90以上のDPSSを採用する
    
    **Return*
    dpss : ndarray [k, npts]
        K個のテーパー群
    eigenvalues : adarray [k]
        K個の固有値 帯域幅のエネルギー集中尺度
    
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
    # 正規化（念のため再確認）
    # -----------------------------------------------------
    vnorm = np.sqrt(np.sum(k_DPSS**2, axis=1))  # 各テーパーのL2ノルム
    k_DPSS = k_DPSS / vnorm[:, None]            # 正規化して単位エネルギーに

    # -----------------------------------------------------
    # 符号統一（positive-standard）
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
    各テーパーごとの固有周波数スペクトル及び固有パワースペクトル
    """
    tapered_data = k_DPSS * data[np.newaxis, :]  # shape: (K, N)
    Jk = np.fft.fft(tapered_data, n=nfft, axis=1) / np.sqrt(fs) 
    Smt_k = np.abs(Jk)**2
    return Jk, Smt_k



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
        p_level:float = 0.05
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



    def MT_Spec(self, data: np.ndarray, fs:float):
        """
        与えられた時系列データ x に対して、
        マルチテーパー (DPSS) 法で PSD を推定する。

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
        
        self.N = len(data)
        # DPSS テーパーの生成 (shape: (K, N))
        self.k_DPSS, self.eigenvalues, self.K = dpss(self.N, self.NW, self.K)

        # # detrend
        # self.data = detrend(data,self.detrend)

        # MT法によるスペクトル推定
        if self.nfft is None:
            self.nfft = len(data)
        self.Jk, self.Smt_k = eigen_psd(self.data, self.k_DPSS, self.fs ,self.nfft) # (K, nfft)
        self.f = np.fft.fftfreq(self.nfft, d=1/self.fs)
        self.Smt = np.mean(self.Smt_k, axis=0)

        # 周波数軸とスペクトル化の場合分け
        if np.iscomplexobj(self.data):
            # 複素信号 → 両側スペクトル
            self.f = np.fft.fftfreq(self.nfft, d=1/self.fs)   # 両側周波数軸
            nfreq = len(self.f)
            self.Smt_k = self.Smt_k[:, :nfreq]
            self.Smt = np.mean(self.Smt_k, axis=0)
        else:
            # 実信号 → 片側スペクトル
            self.f = np.fft.rfftfreq(self.nfft, d=1/self.fs)  # 片側周波数軸
            nfreq = len(self.f)
            self.Smt_k = self.Smt_k[:, :nfreq]
            self.Smt = np.mean(self.Smt_k, axis=0)

        print("self.k_DPS, self.eigenvalues,self.Jk, self.Smt_k, self.Smt, f")

        return None


    def Harmonic_Ftest(self, p_level: float = 0.05):
        """
        Harmonic F-test for detecting significant sinusoidal components
        using Multitaper spectral estimates.

        Parameters
        ----------
        p_level : float
            Significance level (default 0.05)
        """

        npts  = np.shape(self.k_DPSS)[1]    # DPSSターパーの長さ
        kspec = np.shape(self.k_DPSS)[0]    # DPSSターパーの本数

        dof1 = 2
        dof2 = 2 * (kspec - 1)

        # 各テーパーにおける H_k(0) の算出
        H_k0 = (1/self.fs) * np.sum(self.k_DPSS, axis=1)
        H_k0[1::2] = 0  # 奇関数の和は0にする

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

        # 周波数軸に合わせて切り出し
        nfreq = len(self.f)
        F = F[:nfreq]
        p = p[:nfreq]

        self.F_stat = np.zeros((2, nfreq), dtype=float)
        self.F_stat[0, :] = F
        self.F_stat[1, :] = p

        self.F_crit = stats.f.ppf(1 - p_level, dof1, dof2)

        # 有意な周波数を取得
        p_masked = np.where(p > (1 - p_level), p, 0.0)
        local_maxima, _ = signal.find_peaks(p_masked, plateau_size=1)
        nl = len(local_maxima)

        # スペクトル再構成
        self.re_psd = np.zeros((3, nfreq), dtype=float)

        if nl == 0:
            # 有意な線スペクトルがない場合
            self.re_psd[0, :] = self.Smt[:nfreq]
            self.re_psd[1, :] = self.Smt[:nfreq]
            self.re_psd[2, :] = np.zeros(nfreq, dtype=float)
            self.k_psd_back = np.copy(self.Smt_k[:, :nfreq])
        else:
            # 有意な線スペクトルのみ残す
            C_test = np.zeros_like(C)
            C_test[local_maxima] = C[local_maxima]

            H_k = (1/self.fs) * np.fft.fft(self.k_DPSS, n=self.nfft, axis=1)
            back_Jk = np.copy(self.Jk)
            for i in local_maxima:
                jj = (np.arange(self.nfft) - i) % self.nfft
                back_Jk -= C[i] * H_k[:, jj] / np.sqrt(1/self.fs)

            k_psd_back = (np.abs(back_Jk))**2
            re_mt_psd_back = np.mean(k_psd_back, axis=0)
            sline = np.abs(C_test)**2
            re_mt_psd = re_mt_psd_back + sline

            self.k_psd_back = k_psd_back[:, :nfreq]
            self.re_psd[0, :] = re_mt_psd_back[:nfreq]
            self.re_psd[1, :] = re_mt_psd[:nfreq]
            self.re_psd[2, :] = sline[:nfreq]

        # 実信号の場合は片側スペクトルに補正
        if (not np.iscomplexobj(self.data)) and nfreq > 2:
            self.k_psd_back[:, 1:-1] *= 2
            self.re_psd[:, 1:-1] *= 2
            self.F_stat = self.F_stat[:, :nfreq]  # F値とp値も片側に切り出し
            # self.p = self.p[:, :nfreq]  # F値とp値も片側に切り出し

        return None




    # def Harmonic_Ftest(self, p_level):
    #     # Jk < yk
    #     # Vn < DPSS
    #     #Vn0

    #     npts  = np.shape(self.k_DPSS)[1]    # DPSSターパーの長さ (サンプル数)
    #     kspec = np.shape(self.k_DPSS)[0]    # DPSSターパーの本数

    #     C    = np.zeros(self.nfft)
    #     F     = np.zeros(self.nfft)
    #     p     = np.zeros(self.nfft)
    #     dof1 = 2
    #     dof2 = 2 * (kspec - 1)

    #     # 各テーパーにおけるH_k(0)の算出
    #     H_k0 = (1/self.fs)*np.sum(self.k_DPSS, axis=1)  #shape: (k,)
    #     H_k0[1::2] = 0  # 奇関数の和は0にする

    #     # 各周波数における回帰係数 C の算出
    #     H_k0_2sum = np.sum(H_k0**2) 
    #     Jk_Hk0 = np.sum(self.Jk * H_k0[:, np.newaxis], axis=0) #shape (nfft,)
    #     C = np.sqrt(1 / self.fs) * Jk_Hk0 / H_k0_2sum #shape(nfft,)

    #     # F統計量  
    #     # 分子（Fup）の計算
    #     Fup = float(kspec - 1)* H_k0_2sum  * np.abs(C) ** 2  # shape: (nfft,)

    #     # 残差の計算（Fdw）
    #     Jk_hat_1 = (C * H_k0[:, np.newaxis]) / np.sqrt(1/self.fs)
    #     Fdw = (1/self.fs)*np.sum(np.abs(self.Jk - Jk_hat_1 )**2, axis=0)  # shape: (nfft,)

    #     # F値の計算
    #     F = Fup / Fdw  # shape: (nfft,)
    #     F = F[:self.nfft // 2+1]
    #     # p値の計算
    #     p = stats.f.cdf(F, dof1, dof2)  # shape: (nfft,)
    #     p = p[:self.nfft // 2+1]

    #     self.F_stat = np.zeros((2, self.nfft // 2), dtype=float) #(2,nfft)
    #     self.F_stat[0,:] = F
    #     self.F_stat[1,:] = p

    #     self.F_crit = stats.f.ppf(1 - p_level, dof1, dof2)

    #     # 有意な周波数を取得
    #     p[p < (1-p_level)] = 0
    #     local_maxima, _ = signal.find_peaks(p, plateau_size=1)

    #     nl = len(local_maxima)
        
    #     # スペクトル再構成
    #     self.re_psd = np.zeros((3,self.nfft // 2), dtype=float)

    #     if (nl == 0):
    #         self.re_k_psd = self.k_psd
    #         self.re_psd[0,:] = self.mt_psd
    #         self.re_psd[1,:] = self.mt_psd
    #         sline = np.zeros_like(C)
    #         sline = np.abs(sline)**2
    #         self.re_psd[2,:] = sline[:self.nfft // 2]
    #         return None
        
    #     else:
    #         # 検定結果より優位な線スペクトルのみ残す
    #         C_test = np.zeros_like(C)
    #         C_test[local_maxima] = C[local_maxima]  # shape: (nfft,)

    #         # スペクトルの再構成 H_k(f-f1)
    #         H_k = (1/self.fs) * np.fft.fft(self.k_DPSS, n=self.nfft, axis=1)  # shape(k,nfft)
    #         back_Jk = np.copy(self.Jk)
    #         for i in local_maxima:
    #             jj = (np.arange(self.nfft) - i) % self.nfft
    #             back_Jk -= C[i] * H_k[:, jj] / np.sqrt(1/self.fs)

    #         k_psd_back = (np.abs(back_Jk))**2
    #         re_mt_psd_back = np.mean(k_psd_back, axis=0)
    #         sline = np.abs(C_test)**2
    #         re_mt_psd = re_mt_psd_back + sline

    #         # 片側／両側に合わせて切り出し
    #         nfreq = len(self.f)
    #         self.k_psd_back = k_psd_back[:, :nfreq]
    #         self.re_psd = np.zeros((3, nfreq), dtype=float)
    #         self.re_psd[0, :] = re_mt_psd_back[:nfreq]
    #         self.re_psd[1, :] = re_mt_psd[:nfreq]
    #         self.re_psd[2, :] = sline[:nfreq]

    #         # 片側補正（実信号のみ、DC/Nyquist除外）
    #         if (not np.iscomplexobj(self.data)) and nfreq > 2:
    #             self.k_psd_back[:, 1:-1] *= 2
    #             self.re_psd[:, 1:-1] *= 2

    #         return None
