# 0. 準備

## 0-1. 単変量ガウス分布 (§2.3)

### 0-1-1. 最尤推定とベイズ推論 (§2.3.4 - 2.3.6)
- <img src="https://latex.codecogs.com/gif.latex?\displaystyle p(x)=\mathcal{N}(x\:|\:\mu,\sigma^{2}) = \mathcal{N}(x\:|\:\mu,\lambda^{-1})"/>
- $\displaystyle p(x)=\mathcal{N}(x\:|\:\mu,\sigma^{2}) = \mathcal{N}(x\:|\:\mu,\lambda^{-1})$ とおくとき,

    - 最尤推定
        - $\displaystyle\mu_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^{N}x_{n}\;\;\;\left(\mathbb{E}\left[\mu_{\mathrm{ML}}\right]=\mu\right)$
        - $\displaystyle\sigma^{2}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^{N}(x_{n}-\mu_{\mathrm{ML}})^{2}\;\;\;\left(\mathbb{E}\left[\sigma^{2}_{\mathrm{ML}}\right]=\frac{N-1}{N}\sigma^{2}\right)$
        
    - 精度は既知で平均が未知のときのベイズ推論
        - 平均の事前分布: $p(\mu)=\mathcal{N}\left(\mu\:|\:\mu_{0},\lambda_{0}^{-1}\right)$
        - 平均の事後分布: $\displaystyle p(\mu|\mathbf{X})=\mathcal{N}\left(\mu\:\left|\:\frac{N\lambda_{0}^{-1}\mu_{\mathrm{ML}}+\lambda^{-1}\mu_{0}}{N\lambda_{0}^{-1}+\lambda^{-1}},\:\:\left(N\lambda+\lambda_{0}\right)^{-1}\right.\right)$
        - 平均の点推定値: $\displaystyle\mu_{\mathrm{MAP}}=\frac{N\lambda_{0}^{-1}\mu_{\mathrm{ML}}+\lambda^{-1}\mu_{0}}{N\lambda_{0}^{-1}+\lambda^{-1}}$
        
    - 平均は既知で精度が未知のときのベイズ推論       
        - 精度の事前分布: $p(\lambda)=\mathrm{Gam}\left(\lambda\,|\,a_{0},b_{0}\right)$ (ガンマ分布)
        - 精度の事後分布: $\displaystyle p(\lambda|\mathbf{X})=\mathrm{Gam}\left(\lambda\,\left|\,a_{0}+\frac{N}{2},\:\: b_{0}+\frac{1}{2}\sigma^{2}_{\mathrm{ML}}\right.\right)$
        - 精度の点推定値: $\displaystyle \lambda_{\mathrm{MAP}}=\frac{a_{0}+\frac{N}{2}-1}{b_{0}+\frac{1}{2}\sigma^{2}_{\mathrm{ML}}}$
    - 平均も精度も未知のときのベイズ推論       
        - 平均と精度の事前分布: $p(\mu,\lambda\,|\,\mu_{0},\beta,a_{0},b_{0})=\mathcal{N}\left(\mu\:|\:\mu_{0},(\beta\lambda)^{-1}\right)\mathrm{Gam}\left(\lambda\,|\,a_{0},b_{0}\right)$ (ガウス-ガンマ分布)
        - 平均と精度の事後分布: $\displaystyle p(\mu,\lambda\,|\,\mathbf{X},\mu_{0},\beta,a_{0},b_{0})=\mathcal{N}\left(\mu\:\left|\:\mu_{0}+\frac{N}{\beta}\mu_{\mathrm{ML}},\:(\beta\lambda)^{-1}\right.\right)\mathrm{Gam}\left(\lambda\,\left|\,a_{0}+\frac{N}{2},\:b_{0}+\frac{N}{2}\sigma^{2}_{\mathrm{ML}}+N\mu_{\mathrm{ML}}^{2}\right.\right)$

## 0-2. 多変量ガウス分布 (§2.3)
- $\displaystyle p(\mathbf{x})=\mathcal{N}(\mathbf{x}\:|\:\boldsymbol{\mu},\Sigma)=\frac{1}{(2\pi|\Sigma|)^{\frac{D}{2}}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$

### 0-2-1. 共分散行列の対角化 (§2.3)
- $\Sigma$ は対称行列なので正規直交系をなすような固有ベクトルをとれる.
- $\Sigma$ の固有値を $\lambda_{1}, ...,\lambda_{D}$ とし, 固有ベクトルを正規直交系をなすように $\mathbf{u}_{1}, ...,\mathbf{u}_{D}$ ととると $\displaystyle \Sigma=\sum_{i=1}^{D}\lambda_{i}\mathbf{u}_{i}\mathbf{u}_{i}^{T},\:\:\:\Sigma^{-1}=\sum_{i=1}^{D}\frac{1}{\lambda_{i}}\mathbf{u}_{i}\mathbf{u}_{i}^{T}$
    - このガウス分布の密度の等高面は, 中心が $\boldsymbol{\mu}$ の超単位球面を $\mathbf{u}_{i}$ 方向に$\displaystyle\frac{1}{\sqrt{\lambda_{i}}}$倍に引き伸ばしたもの
    
### 0-2-2. モーメント (§2.3)
- $\mathbb{E}\left[\mathbf{x}\right] = \boldsymbol{\mu},\:\:\:\mathbb{E}\left[\mathbf{x}\mathbf{x}^{T}\right] = \boldsymbol{\mu}\boldsymbol{\mu}^{T}+\Sigma$

### 0-2-3. 分割, 条件付き分布, 周辺化 (§2.3.1 - 2.3.2)
- $\displaystyle p\left(\left(
    \begin{array}{c}
        \mathbf{x}_{a}\\
        \mathbf{x}_{b}\\
    \end{array}
    \right)\right)=\mathcal{N}\left(\left(
    \begin{array}{c}
        \mathbf{x}_{a}\\
        \mathbf{x}_{b}\\
    \end{array} 
    \right)
    \:\left|\:\left(
    \begin{array}{c}    
        \boldsymbol{\mu}_{a}\\
        \boldsymbol{\mu}_{b}\\
    \end{array}  
    \right)
    ,
    \left(
    \begin{array}{cc}
        \Sigma_{aa} & \Sigma_{ab} \\
        \Sigma_{ba} & \Sigma_{bb} \\        
    \end{array}  
    \right)
    \right.\right),\:\:\:
    \left(
    \begin{array}{cc}
        \Sigma_{aa} & \Sigma_{ab} \\
        \Sigma_{ba} & \Sigma_{bb} \\        
    \end{array}\right)
    =
    \left(
    \begin{array}{cc}
        \Lambda_{aa} & \Lambda_{ab} \\
        \Lambda_{ba} & \Lambda_{bb} \\        
    \end{array}  
    \right)^{-1}
    $
    とする.
    
    - $\mathbf{x}_{b}$ についての条件付きガウス分布:
    $\displaystyle p(\mathbf{x}_{a}\:|\:\mathbf{x}_{b}) = \mathcal{N}\left(\mathbf{x}_{a}\:|\:\boldsymbol{\mu}_{a}-\Lambda_{aa}^{-1}\Lambda_{ab}(\mathbf{x}_{b}-\boldsymbol{\mu}_{b}),\:\Lambda_{aa}^{-1}\right)$
    - $\mathbf{x}_{b}$ について周辺化した分布:
    $\displaystyle p(\mathbf{x}_{a}) = \mathcal{N}\left(\mathbf{x}_{a}\:|\:\boldsymbol{\mu}_{a},\:\Sigma_{aa}\right)$
    
### 0-2-4. 線形ガウスモデルにおける事前分布, 事後分布 (§2.3.3)
- $\displaystyle p(\mathbf{x})=\mathcal{N}(\mathbf{x}\:|\:\boldsymbol{\mu},\Lambda^{-1}),\:\:\:
p(\mathbf{y}|\mathbf{x})=\mathcal{N}(\mathbf{y}\:|\:\mathbf{A}\mathbf{x}+\mathbf{b},\mathbf{L}^{-1})
$ とおくとき,
    - 同時分布: $\displaystyle p\left(\left(
    \begin{array}{c}
        \mathbf{x}\\
        \mathbf{y}\\
    \end{array}
    \right)\right)=\mathcal{N}\left(\left(
    \begin{array}{c}
        \mathbf{x}\\
        \mathbf{y}\\
    \end{array} 
    \right)
    \:\left|\:\left(
    \begin{array}{c}    
        \boldsymbol{\mu}\\
        \mathbf{A}\boldsymbol{\mu}+\mathbf{b}\\
    \end{array}  
    \right)
    ,
    \left(
    \begin{array}{cc}
        \Lambda^{-1} & \Lambda^{-1}\mathbf{A}^{T} \\
        \mathbf{A}\Lambda^{-1} & \mathbf{L}^{-1}+\mathbf{A}\Lambda^{-1}\mathbf{A}^{T} \\        
    \end{array}  
    \right)
    \right.\right) 
    $
        - ただし $
        \displaystyle
        \left(
        \begin{array}{cc}
            \Lambda^{-1} & \Lambda^{-1}\mathbf{A}^{T} \\
            \mathbf{A}\Lambda^{-1} & \mathbf{L}^{-1}+\mathbf{A}\Lambda^{-1}\mathbf{A}^{T} \\        
        \end{array}  
        \right) =
        \left(
        \begin{array}{cc}
            \Lambda+\mathbf{A}^{T}\mathbf{L}\mathbf{A} & -\mathbf{A}^{T}\mathbf{L} \\
            -\mathbf{L}\mathbf{A} & \mathbf{L} \\        
        \end{array}  
        \right)^{-1}
        $
    - 周辺分布: $\displaystyle p(\mathbf{y})=\mathcal{N}(\mathbf{y}\:|\:\mathbf{A}\boldsymbol{\mu}+\mathbf{b},\:\:\mathbf{L}^{-1}+\mathbf{A}\Lambda^{-1}\mathbf{A}^{T})$
    - 条件付き分布: $\displaystyle p(\mathbf{x}|\mathbf{y})=\mathcal{N}(\mathbf{x}\:|\:(\Lambda+\mathbf{A}^{T}\mathbf{L}\mathbf{A})^{-1}(\mathbf{A}^{T}\mathbf{L}(\mathbf{y}-\mathbf{b})+\Lambda\boldsymbol{\mu}),\:\:(\Lambda+\mathbf{A}^{T}\mathbf{L}\mathbf{A})^{-1})$
    
### 0-2-5. 最尤推定とベイズ推論 (§2.3.4 - 2.3.6)

- $\displaystyle p(\mathbf{x})=\mathcal{N}(\mathbf{x}\:|\:\boldsymbol{\mu},\Sigma) = \mathcal{N}(\mathbf{x}\:|\:\boldsymbol{\mu},\Lambda^{-1})$ とおくとき,

    - 最尤推定
        - $\displaystyle\boldsymbol{\mu}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_{n}\;\;\;\left(\mathbb{E}\left[\boldsymbol{\mu}_{\mathrm{ML}}\right]=\boldsymbol{\mu}\right)$
        - $\displaystyle\boldsymbol{\Sigma}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^{N}(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}})^{T}\;\;\;\left(\mathbb{E}\left[\boldsymbol{\Sigma}_{\mathrm{ML}}\right]=\frac{N-1}{N}\boldsymbol{\Sigma}\right)$
        
    - 精度行列は既知で平均が未知のときのベイズ推論
        - 平均の事前分布: $p(\boldsymbol{\mu})=\mathcal{N}\left(\boldsymbol{\mu}\:|\:\boldsymbol{\mu}_{0},\boldsymbol{\Lambda}_{0}^{-1}\right)$
        - 平均の事後分布: $p(\boldsymbol{\mu}|\mathbf{X})=\mathcal{N}\left(\boldsymbol{\mu}\:|\:\left(N\boldsymbol{\Lambda}+\boldsymbol{\Lambda}_{0}\right)^{-1}\left(N\boldsymbol{\Lambda}\boldsymbol{\mu}_{\mathrm{ML}}+\boldsymbol{\Lambda}_{0}\boldsymbol{\mu}_{0}\right),\:\left(N\boldsymbol{\Lambda}+\boldsymbol{\Lambda}_{0}\right)^{-1}\right)$
        - 平均の点推定値: $\boldsymbol{\mu}_{\mathrm{MAP}}=\left(N\boldsymbol{\Lambda}+\boldsymbol{\Lambda}_{0}\right)^{-1}\left(N\boldsymbol{\Lambda}\boldsymbol{\mu}_{\mathrm{ML}}+\boldsymbol{\Lambda}_{0}\boldsymbol{\mu}_{0}\right)$
        
    - 平均は既知で精度行列が未知のときのベイズ推論       
        - 精度行列の事前分布: $p(\boldsymbol{\Lambda})=\mathcal{W}\left(\boldsymbol{\Lambda}\:|\:\mathbf{W},\nu\right)$ (ウィシャート分布)
        - 精度行列の事後分布: $\displaystyle p(\boldsymbol{\Lambda}|\mathbf{X})=\mathcal{W}\left(\boldsymbol{\Lambda}\:\left|\:\left(\mathbf{W}^{-1}+\sum_{n=1}^{N}(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_{n}-\boldsymbol{\mu}_{\mathrm{ML}})^{T}\right)^{-1},\:\:\nu+N\right.\right)$
    - 平均も精度行列も未知のときのベイズ推論       
        - 平均と精度行列の事前分布: $p(\boldsymbol{\mu},\boldsymbol{\Lambda}\,|\,\boldsymbol{\mu}_{0},\beta,\mathbf{W},\nu)=\mathcal{N}\left(\boldsymbol{\mu}\:|\:\boldsymbol{\mu}_{0},(\beta\boldsymbol{\Lambda})^{-1}\right)\mathcal{W}\left(\boldsymbol{\Lambda}\:|\:\mathbf{W},\nu\right)$ (ガウス-ウィシャート分布)


# 1. 2クラス分類のための識別関数

## 1-1. 一般線形モデル (§4)

- $y=f(\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x}))$ とおいて, $\mathbf{w}$を最適化する. $f$は活性化関数.
    - $f$ は線形でも非線形でもよい.
    - 基底関数 $\boldsymbol{\phi}$ はあらかじめ決めておき, 動かさずに固定する.

### 1-1-1. 一般線形モデル > 識別関数 > 最小二乗法 (§4.1.1 - 4.1.3)
- 二乗和誤差関数 $\displaystyle E(W)=\frac{1}{2}\mathrm{Tr}\{(XW-T)^{T}(XW-T)\}$ を最小化する.
- 最適解は $W=(X^{T}X)^{-1}X^{T}T = X^{\dagger}T$
- $W\mathbf{x}\ge0$のとき$C_{1}$, $W\mathbf{x}<0$のとき$C_{0}$と判定する.
    - あまりうまくいかない
    - そもそも最小二乗法とは目的変数の条件付き確率分布にガウス分布を仮定したときの最尤推定
    - ここでは目的変数は二値変数であり, ガウス分布とはかけ離れているので当然のこと

### 1-1-2. 一般線形モデル > 識別関数 > フィッシャーの線形判別 (§4.1.4 - 4.1.6)
- フィッシャーの判別規準を最大化する.
- 最適解は $\mathbf{w}\propto (\mathbf{m}_{1}-\mathbf{m}_{0})$
- $\mathbf{w}^{T}\mathbf{x}\ge -w_{0}$のとき$C_{1}$, $\mathbf{w}^{T}\mathbf{x}<-w_{0}$のとき$C_{0}$と判定する.
    - しきい値$w_{0}$は$p(y|C_{k})$をモデル化して最尤推定などで求める
    
## 1-2. 決定木モデル (§14.4)


# 2. 2クラス分類のための識別モデルと生成モデル
目標値が $t\in\{0,1\}$ であり, $y(\boldsymbol{\phi}(\mathbf{x}), \mathbf{w})$ が $p(t=1|\boldsymbol{\phi}(\mathbf{x}))$ を出力するような2クラス分類モデルを構成したいとき:

## 2-1. 一般線形モデル (§4)

- $y=f(\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x}))$ とおいて, $\mathbf{w}$を最適化する. $f$は活性化関数.
    - $f$ には非線形なものをえらぶ.
    - 基底関数 $\boldsymbol{\phi}$ はあらかじめ決めておき, 動かさずに固定する.

### 2-1-1. 一般線形モデル > 識別モデル (§4.3)

#### 2-1-1-1. 一般線形モデル > 識別モデル > ロジスティック回帰  (§4.3.1 - 4.3.3)
- $f$はロジスティックシグモイド. 理由は誤差関数の勾配が簡潔に書けるため(正準連結関数).
- 最尤推定する場合の誤差関数(負の対数尤度比)はcross-entropy: $\displaystyle E(\mathbf{w})=-\sum_{n=1}^{N}\left(t_{n}\ln y_{n}+(1-t_{n})\ln(1-y_{n})\right)$
    - この $E(\mathbf{w})$ は解析的には最小化できない(逐次的に小さくするしかない).
    - $E(\mathbf{w})$ の勾配は $\displaystyle\nabla_{\mathbf{w}}E(\mathbf{w})=\sum_{n=1}^{N}(y_{n}-t_{n})\boldsymbol{\phi}$
    - ニュートン・ラフソン法によって$\mathbf{w}$を更新する方法は $\mathbf{w}^{\mathrm{new}}=\mathbf{w}^{\mathrm{old}}-H^{-1}\nabla_{\mathbf{w}}E(\mathbf{w})=(\Phi^{T}R\Phi)^{-1}\Phi^{T}R\mathbf{z}$
        - $\Phi$ は $\boldsymbol{\phi}_{n}^{T}$ を行ベクトルにもつ$N\times D$行列, $R$は $R_{nn}=y_{n}(1-y_{n})$ なる対角行列
        - $\mathbf{z}$ は $N$次元ベクトルで $\mathbf{z}=\Phi \mathbf{w}^{\mathrm{old}} - R^{-1}(\textsf{y}-\textsf{t})$
    - 実際には正則化項を追加しないと過学習するので注意.

#### 2-1-1-2, 一般線形モデル > 識別モデル > プロビット回帰  (§4.3.5)

#### 2-1-1-3. 一般線形モデル > 識別モデル > ベイズロジスティック回帰  (§4.5)

### 2-1-2. 一般線形モデル >  確率的生成モデル (§4.2)
- $f$はロジスティックシグモイド.
- 理由は $\displaystyle p(t=1|\mathbf{x})=\displaystyle p(C_{1}|\mathbf{x})=\frac{1}{1+\displaystyle\frac{p(\mathbf{x}|C_{1})p(C_{1})}{p(\mathbf{x}|C_{0})p(C_{0})}}=\sigma(a),\:\:\:\:a=\ln\frac{p(\mathbf{x}|C_{1})p(C_{1})}{p(\mathbf{x}|C_{0})p(C_{0})}$ とおけるため.
- $a$を構成する事前確率を推定して線形モデルに書き直せばよい.

#### 2-1-2-1. 一般線形モデル > 確率的生成モデル > 入力が連続値の場合  (§4.2.1 - 4.2.2)

- 一般に $p(\mathbf{x}|C_{k})$ が正準形指数型分布族のメンバー $(p(\mathbf{x}|\boldsymbol\lambda_{k})=h(\mathbf{x})g(\boldsymbol\lambda_{k})\exp(\boldsymbol\lambda_{k}^{T}\mathbf{x}))$ であるとき
    - $s$をクラス間で共有された尺度パラメーターとすると $a=\displaystyle\frac{1}{s}(\boldsymbol\lambda_{1}-\boldsymbol\lambda_{0})^{T}\mathbf{x}+\ln\frac{g(\boldsymbol\lambda_{1})p(C_{1})}{g(\boldsymbol\lambda_{0})p(C_{0})}$
    - つまり $\displaystyle\mathbf{w} = \displaystyle\frac{1}{s}(\boldsymbol\lambda_{1}-\boldsymbol\lambda_{0}),\:\:\:\: w_{0}=\ln\frac{g(\boldsymbol\lambda_{1})p(C_{1})}{g(\boldsymbol\lambda_{0})p(C_{0})}$
    
        - あとは $\boldsymbol\lambda_{1},\:\boldsymbol\lambda_{0},\:p(C_{1}),\:p(C_{0})$ を推定すればよい
  
  
- とくに $p(\mathbf{x}|C_{k})$ がガウス分布 $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_{k},\Sigma)$ のとき

    - $\displaystyle a=\left(\Sigma^{-1}(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{0})\right)^{T}\mathbf{x}-\frac{1}{2}\left(
\boldsymbol{\mu}_{1}^{T}\Sigma^{-1}\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{0}^{T}\Sigma^{-1}\boldsymbol{\mu}_{0}
\right)
+
\ln \frac{p(C_{1})}{p(C_{0})}$

    - つまり $\displaystyle\mathbf{w} = \Sigma^{-1}(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{0}),\:\:\:\:
w_{0} = -\frac{1}{2}\left(
\boldsymbol{\mu}_{1}^{T}\Sigma^{-1}\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{0}^{T}\Sigma^{-1}\boldsymbol{\mu}_{0}
\right)
+
\ln \frac{p(C_{1})}{p(C_{0})}$

        - 最尤推定解は $\displaystyle p(C_{k})=\frac{N_{k}}{N},\:\:\:\boldsymbol{\mu}_{k}=\frac{1}{N_{k}}\sum_{n=1}^{N}t_{n}\mathbf{x}_{k},\:\:\:\Sigma=\frac{1}{N}\sum_{k}\sum_{n\in C_{k}}(\mathbf{x}_{k}-\boldsymbol{\mu}_{k})(\mathbf{x}_{k}-\boldsymbol{\mu}_{k})^{T}$
        - MAP推定解は $\displaystyle p(C_{k})=\frac{N_{k}(N_{k}+1)}{N(N+1)}$
  
  
- とくに $p(\mathbf{x}|C_{k})$ がガウス分布であり, しかも各成分が独立であるとき $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_{k},\Sigma), \:\:\sigma_{ij}=0 \:\mathrm{for}\: i\ne j$  
👉`sklearn.naive_bayes.GaussianNB`

#### 2-1-2-2. 一般線形モデル > 確率的生成モデル > 入力が離散値の場合  (§4.2.3)

- 特徴値$x_{i}$が$(0,1)$の離散値をとり, $i\ne j$に対して$x_{i},x_{j}$が独立なとき $\displaystyle \left( p(\mathbf{x}|C_{k})=\prod_{i=1}^{D} \mu_{ki}^{x_{i}}(1-\mu_{ki})^{1-x_{i}}\right)$  
👉`sklearn.naive_bayes.BernoulliNB`  
    - $\displaystyle a = \sum_{i=1}^{D}\ln\frac{\mu_{1i}(1-\mu_{0i})}{\mu_{0i}(1-\mu_{1i})}x_{i}+\sum_{i=1}^{D}\ln\frac{1-\mu_{1i}}{1-\mu_{0i}}+\ln\frac{p(C_{1})}{p(C_{0})}$
    - つまり $\displaystyle\mathbf{w} = \biggl\{\ln\frac{\mu_{1i}(1-\mu_{0i})}{\mu_{0i}(1-\mu_{1i})}\biggr\},\:\:\:\:w_{0}=\sum_{i=1}^{D}\ln\frac{1-\mu_{1i}}{1-\mu_{0i}}+\ln\frac{p(C_{1})}{p(C_{0})}$
        - 最尤推定解は $\displaystyle p(C_{k})=\frac{N_{k}}{N},\:\:\:\:\mu_{ki}=\frac{N_{ki}}{N_{k}}$
        - MAP推定解は $\displaystyle p(C_{k})=\frac{N_{k}(N_{k}+1)}{N(N+1)},\:\:\:\:\mu_{ki}=\frac{N_{ki}(N_{ki}+1)}{N_{k}(N_{k}+1)}$
        
        
- 特徴値$x_{i}$が比例尺度であって, 0以上の整数の離散値をとり, $i\ne j$に対して$x_{i},x_{j}$が独立なとき  
👉`sklearn.naive_bayes.MultinomialNB`, `sklearn.naive_bayes.ComplementNB`

- 特徴値$x_{i}$が$M_{i}$次元のone-hot vector $\boldsymbol{\phi}_{i}$で表現でき, $i\ne j$に対して$x_{i},x_{j}$が独立なとき $\displaystyle \left( p(\mathbf{x}|C_{k})=\prod_{i=1}^{D}\prod_{j=1}^{M_{i}} \mu_{kij}^{\phi_{ij}} \right)$

    - $\displaystyle a = \sum_{i=1}^{D}\sum_{j=1}^{M_{i}}\ln\frac{\mu_{1ij}}{\mu_{0ij}}\phi_{ij}+\ln\frac{p(C_{1})}{p(C_{0})} = \sum_{i=1}^{D}\mathbf{m}_{i}^{T}\boldsymbol{\phi}_{i}+\ln\frac{p(C_{1})}{p(C_{0})}\:\:\:\:\left(\mathbf{m}_{i}=\biggl\{ \ln\frac{\mu_{1ij}}{\mu_{0ij}}\biggr\}\right)$
    - つまり $\displaystyle\mathbf{w} = \biggl\{\mathbf{m}_{i}\biggr\},\:\:\:\:w_{0}=\ln\frac{p(C_{1})}{p(C_{0})}$
        - 最尤推定解は $\displaystyle p(C_{k})=\frac{N_{k}}{N},\:\:\:\:\mu_{kij}=\frac{N_{kij}}{N_{ki}}$
        - MAP推定解は $\displaystyle p(C_{k})=\frac{N_{k}(N_{k}+1)}{N(N+1)},\:\:\:\:\mu_{kij}=\frac{N_{kij}(N_{kij}+1)}{N_{ki}(N_{ki}+1)}$
        
        
## 2-2. フィードフォワードネットワーク (§5)

- $y=f(\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x}))$ とおいて, $\mathbf{w}$と$\boldsymbol{\phi}$を最適化する(基底関数も動かす). $f$は非線形活性化関数.


# 3. 2クラス分類のための識別モデルと生成モデル

目標値が $t\in\{-1,1\}$ であり, $y(\mathbf{x}, \mathbf{w})$ が $-1\le y \le 1$ を出力するような2クラス分類モデルを構成したいとき:

- 活性化関数をロジスティックシグモイドから $\tanh$ に変更する.


# 4. 多クラス分類のための識別関数

# 5. 多クラス分類のための識別モデルと生成モデル

# 6. 多クラス分類のための識別モデルと生成モデル

# 7. 目標値がスカラーのときの回帰
## 7-1. 一般線形モデル
- $y=\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x})$ とおいて, $\mathbf{w}$を最適化する.
    - 基底関数 $\boldsymbol{\phi}$ はあらかじめ決めておき, 動かさずに固定する.
    - 基底関数を $\boldsymbol{\phi}_{j}(\mathbf{x})=x^{j}$ とすれば多項式フィッテイングとなる.

### 7-1. 一般線形モデル > 予測値を点推定
- 二乗和誤差関数 $\displaystyle E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}(t_{n}-\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x}))^{2}$ を最小化する.
- 最適解は $\mathbf{w}=\Phi^{\dagger}\textsf{t}$
- 予測値は $t=(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x}$
    - $\Phi^{\dagger}$ は $\Phi$ のムーア・ペンローズ擬似逆行列で, $\Phi^{\dagger}=(\Phi^{T}\Phi)^{-1}\Phi^{T}$
    - $\textsf{t}$ は $N$ 次元ベクトルで, $\textsf{t}=(t_{1} ... t_{n})^{T}$

### 7-2. 一般線形モデル > 予測値を確率的に推定
#### 7-2-1. 一般線形モデル > 予測値を確率的に推定 > 頻度主義
- 目標値 $t$ にガウスノイズを仮定する $\displaystyle \left(t=\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x})+\epsilon,\;\;p(\epsilon)=\mathcal{N}(\epsilon|0,\beta^{-1})\right)$.
- 次の(1)(2)の結論は同じ:
    - (1) 最尤推定する.
    - (2) 二乗和誤差関数 $\displaystyle E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}(t_{n}-\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x}))^{2}$を最小化する.
- 最適解は $\displaystyle\mathbf{w}_{\mathrm{ML}}=\Phi^{\dagger}\textsf{t},\:\:\:\:\beta^{-1}_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}(t_{n}-\mathbf{w}^{T}_{\mathrm{ML}}\boldsymbol{\phi}(\mathbf{x}))^{2}=\frac{1}{N}\sum_{n=1}^{N}(t_{n}-(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x})^{2}$
- 予測分布は $\displaystyle p(t|\mathbf{x},\mathbf{w}_{\mathrm{ML}},\beta^{-1}_{\mathrm{ML}})=\mathcal{N}(t|w_{\mathrm{ML}}^{T}\mathbf{x},\beta^{-1}_{\mathrm{ML}}) = \mathcal{N}\left(t\:\left|\:(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x}, \frac{1}{N}\sum_{n=1}^{N}(t_{n}-(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x})^{2}\right.\right)$
- 回帰関数は $(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x}=\mathbb{E}_{t}\left[t|\mathbf{x}\right]$ をみたす.
    - $\Phi^{\dagger}$ は $\Phi$ のムーア・ペンローズ擬似逆行列で, $\Phi^{\dagger}=(\Phi^{T}\Phi)^{-1}\Phi^{T}$
    - $\textsf{t}$ は $N$ 次元ベクトルで, $\textsf{t}=(t_{1} ... t_{n})^{T}$
    - $\beta^{-1}_{\mathrm{ML}}$ は回帰関数まわりでの残差分散となっている.
    
#### 7-2-2. 一般線形モデル > 予測値を確率的に推定 > ベイズ線形回帰
- 目標値 $t$ にガウスノイズを仮定する $\displaystyle \left(t=\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x})+\epsilon,\;\;p(\epsilon)=\mathcal{N}(\epsilon|0,\beta^{-1})\right)$.
- 次の(1)(2)の結論は同じ:
    - (1) $\mathbf{w}$ の事前分布を $p(\mathbf{w}|\alpha)=\mathcal{N}(\mathbf{w}|\mathbf{0},\alpha^{-1}I)$ とおいてMAP推定する.
    - (2) L2ノルムで正則化した二乗和誤差関数 $\displaystyle E(\mathbf{w},\lambda)=\frac{1}{2}\sum_{n=1}^{N}(t_{n}-\mathbf{w}^{T}\boldsymbol{\phi}(\mathbf{x}))^{2} + \frac{\lambda}{2}\sum_{j=1}^{D}w_{j}^{2}$ を最小化する.
        - (1)は(2)で $\displaystyle\lambda = \frac{\alpha}{\beta}$ とおいたものに帰着する.
- 最適解は $\displaystyle\mathbf{w}_{\mathrm{MAP}}=(\Phi^{\dagger}+\lambda I)\textsf{t}$
- 予測分布は $\displaystyle p(t|\textsf{t},\mathbf{x},\alpha,\beta)=\mathcal{N}(t|w_{\mathrm{ML}}^{T}\mathbf{x},\beta^{-1}_{\mathrm{ML}}) = \mathcal{N}\left(t\:\left|\:(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x}, \frac{1}{N}\sum_{n=1}^{N}(t_{n}-(\Phi^{\dagger}\textsf{t})^{T}\mathbf{x})^{2}\right.\right)$

# 8. 目標値がベクトルのときの回帰
## 8-1. 一般線形モデル
- $y=W^{T}\boldsymbol{\phi}(\mathbf{x})$ とおいて, $W$を最適化する.
    - 基底関数 $\boldsymbol{\phi}$ はあらかじめ決めておき, 動かさずに固定する.

### 8-1. 一般線形モデル > 目標値を決定論的に扱う
- 目標値 $t$ を決定論的に扱うとき,
    - 二乗和誤差関数 $\displaystyle E(W)=\frac{1}{2}\mathrm{Tr}\{(XW-T)^{T}(XW-T)\}$ を最小化する.
