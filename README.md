# Lunar Lander PPO 強化学習プロジェクト

このプロジェクトでは、強化学習アルゴリズム **PPO（Proximal Policy Optimization）** を使用して、`LunarLander-v3` 環境で着陸制御エージェントを学習させます。  
ライブラリには [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) と [Gymnasium](https://gymnasium.farama.org/) を使用しています。


## 🎯 プロジェクトの目的

月面着陸を模した `LunarLander` 環境において、強化学習エージェントが安全に着陸する動作を自動的に獲得できるように学習させます。


## 📁 プロジェクト構成
```
.
├── README.md                  # 本ドキュメント
├── environment.yml            # Conda 環境ファイル
├── train_lunar_lander.py     # 学習スクリプト（PPOで訓練）
├── evaluate_lunar_lander.py  # 学習済みモデルの評価スクリプト
├── spot_simulation.gif       # 評価結果のGIF（README用に埋め込み可能）
├── lunar-lander-ppo-episode-5.mp4  # 評価結果の動画（高画質保存）

```

## 🛠️ 実装手順

以下の手順をコマンドプロンプトで行ってください。

### 1. リポジトリのクローン

```bash
git clone https://github.com/Kingkazu106/lunar_lander_RL.git
cd lunar_lander_RL
```
### 2. 環境の構築
このプロジェクトには `environment.yml` が付属しています。

```bash
conda env create -f environment.yml
conda activate lunar_lander_env
```

### 3. 学習の実行（モデルの訓練）
```bash
python train_lunar_lander.py
```
学習済みモデルは `ppo_lunar_lander.zip` として保存されます。
学習ログは `lunar_lander_tensorboard/` に出力されます。

### 4. 学習済みモデルの評価（動作確認）
```bash
python evaluate_lunar_lander.py
```
評価結果の動画`lunar-lander-ppo-episode-5.mp4`が生成されます。
## 🎥出力例

![Spot Simulation](https://github.com/Kingkazu106/lunar_lander_RL/blob/main/spot_simulation.gif)


# 🌌 PPO による LunarLander の強化学習

このプロジェクトでは、強化学習アルゴリズム「Proximal Policy Optimization（PPO）」を用いて、LunarLander 環境におけるエージェントの制御を学習させます。



## 📘 背景：マルコフ決定過程（MDP）

強化学習は、**マルコフ決定過程（MDP）** に基づいてモデル化されます。

- 状態空間：$s \in \mathcal{S}$
- 行動空間：$a \in \mathcal{A}$
- 遷移確率：$P(s' \mid s, a)$
- 報酬関数：$r(s, a)$
- 割引率：$\gamma \in [0, 1)$

目的は、**期待累積報酬**を最大化する方策 $\pi(a \mid s)$ を学習することです：

$$
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$



## 🤖 PPO（Proximal Policy Optimization）

PPOは、方策勾配法において安定性と効率を両立したアルゴリズムです。  
従来の方策 $\pi_{\theta_{old}}$ と新しい方策 $\pi_\theta$ の比率：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

PPOの目的関数は、以下のクリップ付き損失関数を最大化します：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ 
\min\left( 
r_t(\theta) \hat{A}_t, 
\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t 
\right) 
\right]
$$

ここで、$\hat{A}_t$ は Advantage（価値関数と実報酬の差）です。

---

## ⚙️ 学習の設定

```python
env = make_vec_env('LunarLander-v3', n_envs=16)

model = PPO(
    'MlpPolicy', 
    env, 
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)
