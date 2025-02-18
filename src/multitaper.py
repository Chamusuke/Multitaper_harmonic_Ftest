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
        2*nw異能の場合,固有値0.90以上のDPSSを採用する
    
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
        valid_indices = np.where(eigenvalues >= 0.90)[0]
        k_DPSS = k_DPSS[valid_indices]
        eigenvalues = eigenvalues[valid_indices]
        k = len(valid_indices)
        print(f"A unique taper of eigenvalues (0.8<) was selected. K={k}")

    return k_DPSS, eigenvalues, k

def eigen_psd(data, k_DPSS, fs, nfft):
    """
    各テーパーごとの固有周波数スペクトル及び固有パワースペクトル
    (Eigen Spectrum and PSD of each DPSS)
    Jk : 固有周波数スペクトル密度 (Eigen Spectrum)
    Sk : 固有パワースペクトル密度 (Eigen PSD)
    """
    tapered_data = k_DPSS * data[np.newaxis, :]  # shape: (K, N)
    Jk = np.sqrt(1/fs)*np.fft.fft(tapered_data, n=nfft, axis=1)  # shape: (K, nfft)
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
        self.k_DPSS = None
        self.f = None
        self.fs = None
        self.mt_psd = None
        self.re_psd = None


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
        
        self.N = len(data)
        # DPSS テーパーの生成 (shape: (K, N))
        self.k_DPSS, self.eigenvalues, self.K = dpss(self.N, self.NW, self.K)

        # detrend
        self.data = detrend(data,self.detrend)

        # MT法によるスペクトル推定
        if self.nfft is None:
            self.nfft = len(data)

        self.Jk, k_psd = eigen_psd(self.data, self.k_DPSS, self.fs ,self.nfft) # (K, nfft)
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
        # Jk < yk
        # Vn < DPSS
        #Vn0

        npts  = np.shape(self.k_DPSS)[1]    # DPSSターパーの長さ (サンプル数)
        kspec = np.shape(self.k_DPSS)[0]    # DPSSターパーの本数

        C    = np.zeros(self.nfft)
        F     = np.zeros(self.nfft)
        p     = np.zeros(self.nfft)
        dof1 = 2
        dof2 = 2 * (kspec - 1)

        # 各テーパーにおけるH_k(0)の算出
        H_k0 = (1/self.fs)*np.sum(self.k_DPSS, axis=1)  #shape: (k,)
        H_k0[1::2] = 0  # 奇関数の和は0にする

        # 各周波数における回帰係数 C の算出
        H_k0_2sum = np.sum(H_k0**2) 
        Jk_Hk0 = np.sum(self.Jk * H_k0[:, np.newaxis], axis=0) #shape (nfft,)
        C = np.sqrt(1 / self.fs) * Jk_Hk0 / H_k0_2sum #shape(nfft,)

        # F統計量  
        # 分子（Fup）の計算
        Fup = float(kspec - 1)* H_k0_2sum  * np.abs(C) ** 2  # shape: (nfft,)

        # 残差の計算（Fdw）
        Jk_hat_1 = (C * H_k0[:, np.newaxis]) / np.sqrt(1/self.fs)
        Fdw = (1/self.fs)*np.sum(np.abs(self.Jk - Jk_hat_1 )**2, axis=0)  # shape: (nfft,)

        # F値の計算
        F = Fup / Fdw  # shape: (nfft,)
        F = F[:self.nfft // 2]
        # p値の計算
        p = stats.f.cdf(F, dof1, dof2)  # shape: (nfft,)
        p = p[:self.nfft // 2]

        self.F_stat = np.zeros((2, self.nfft // 2), dtype=float) #(2,nfft)
        self.F_stat[0,:] = F
        self.F_stat[1,:] = p

        self.F_crit = stats.f.ppf(1 - p_level, dof1, dof2)

        # 有意な周波数を取得
        p[p < (1-p_level)] = 0
        local_maxima = signal.argrelextrema(p, np.greater)[0]  # 局所最大値のインデックスを取得

        nl = len(local_maxima)
        
        # スペクトル再構成
        self.re_psd = np.zeros((3,self.nfft // 2), dtype=float)

        if (nl == 0):
            self.re_k_psd = self.k_psd
            self.re_psd[0,:] = self.mt_psd
            self.re_psd[1,:] = self.mt_psd
            self.re_psd[2,:] = sline[:self.nfft // 2]
            return None
        
        else:
            #検定結果より優位な線スペクトルのみ残す
            C_test = np.zeros_like(C)
            C_test[local_maxima] = C[local_maxima] #shape: (nfft,)

            #スペクトルの再構成 H_k(f-f1) ただし(f1-W<f<F1+W )
            H_k = (1/self.fs)*np.fft.fft(self.k_DPSS, n=self.nfft, axis=1) #shape(k,nfft)
            back_Jk = np.copy(self.Jk)
            for s in range(nl):
                i = local_maxima[s]  # ピーク位置
                jj = (np.arange(self.nfft) - i) % self.nfft  # ベクトル化（負の値を補正） (nfft,)
                back_Jk = back_Jk - C[i] * H_k[:, jj] / np.sqrt(1/self.fs) # ループなしでブロードキャスト計算
  
            k_psd_back = (np.abs(back_Jk))**2 #shape: (k,nfft)
            re_mt_psd_back = np.mean(k_psd_back, axis=0)  #shape:(nfft,)
            sline = np.abs(C_test)**2
            re_mt_psd = sline + re_mt_psd_back #shape:(nfft,)

            self.k_psd_back = k_psd_back[:,:self.nfft // 2]
            self.re_psd[0,:] = re_mt_psd_back[:self.nfft // 2]  
            self.re_psd[1,:] = re_mt_psd[:self.nfft // 2] 
            self.re_psd[2,:] = sline[:self.nfft // 2] 

            self.k_psd_back[:, 1:-1] *= 2  
            self.re_psd[:,1:-1] *= 2 

            return None
