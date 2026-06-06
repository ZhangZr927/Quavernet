# QuaverNet: AI-Powered 4K Chart Generator (AI 音游谱面生成器)

[English](#english) | [中文](#中文)

---

<h2 id="english">English</h2>

### 📌 Project Purpose
QuaverNet is a Deep Learning project designed to automatically generate high-quality 4K rhythm game charts for **Quaver** (or any VSRG with similar formats) from raw audio files (`.mp3`). 

Unlike traditional rule-based generators that often produce robotic or out-of-sync patterns, QuaverNet mimics human charting styles. It can accurately capture beats, vocals, and dynamic drops, and arrange them into playable, expressive, and difficulty-adjustable patterns.

### ✨ Features & Results
*   **Insane Rhythm Accuracy:** Precisely aligned to a 1/48 beat grid. It captures not only drum kicks but also subtle vocal drops.
*   **Dynamic Difficulty Control:** You can set the target difficulty (e.g., 2.5, 4.0, 7.0). The model dynamically scales the note density and chord usage (jumps, hands, quads) accordingly.
*   **Human-like Patterns:** Generates natural streams, jacks, and chords, avoiding physically impossible/unplayable combinations.

### 🚀 Quick Start (Inference Only)
If you only want to generate charts and do not want to train the model, you **ONLY** need to download the following files:
1.  `stage2_infer.py` (The main execution script)
2.  `stage2_model.py` & `stage2_config.py`
3.  `init_model.py` & `init_config.py`
4.  `requirements.txt`
5.  **Model Weights:** `best_rhythm_model.pth` and `best_stage2_generator.pth` (Download these from the GitHub [Releases] page).

**Usage:**
1. Install dependencies using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, install manually: `pip install torch numpy scipy librosa pyyaml tqdm`)*
2. Place your `.mp3` file (e.g., `song.mp3`) in the same directory.
3. Open `stage2_infer.py`, scroll to the bottom, and modify the `TEST_MP3` and `TEST_TARGET_DIFF` variables:
   ```python
   TEST_MP3 = "song.mp3"
   TEST_TARGET_DIFF = 4.5  # Adjust difficulty here
   ```
4. Run the script:
   ```bash
   python stage2_infer.py
   ```
5. Check the `ai_2stage_chart` folder for your generated `.qua` file! Import it into Quaver and enjoy.

### 🧠 Model Architecture
The project utilizes a highly decoupled **Two-Stage Generation Pipeline**:
*   **Stage 1: RhythmNet (When to hit):** A ResNet-1D + BiLSTM + Multi-Head Attention architecture. It incorporates **FiLM (Feature-wise Linear Modulation)** layers to deeply condition the network on the target difficulty. It outputs a highly accurate 1D probability sequence representing the rhythm density.
*   **Stage 2: Masked Pattern Generator (Where to hit):** A U-Net 1D style Generator trained via WGAN-GP (Wasserstein GAN with Gradient Penalty). It takes the audio, difficulty, and the **rhythm mask** from Stage 1 as inputs. A physical masking constraint and rejection penalty are applied so the model focuses purely on lane distribution without ruining the rhythm.

### ⚠️ Disclaimer
1. **Academic & Personal Use Only:** This project is developed purely for academic research, deep learning exploration, and personal entertainment. 
2. **Music Copyright:** The audio files (`.mp3`) processed by this tool are the property of their respective copyright holders. Please ensure you have the legal right to use the audio files. The author of this repository does not own the rights to any music used to test or train this AI.
3. **Game Copyright:** "Quaver" and its related assets are trademarks and properties of the Quaver development team. This project is unofficial and is not affiliated with, endorsed by, or connected to the Quaver team in any way.
4. **Training Data:** The deep learning models were trained on community-created charts. We deeply respect the human charters whose creativity made this research possible. The AI is meant to be a supplementary tool for generating draft charts, not a replacement for the human charting community.
5. **No Liability:** The repository owner shall not be held liable for any copyright disputes, account bans, or other consequences arising from users uploading AI-generated charts to official game servers or public ranking systems. Use responsibly.

---

<h2 id="中文">中文</h2>

### 📌 项目目的
QuaverNet 是一个基于深度学习的 AI 谱面生成项目。它的核心目的是：输入任意一首 `.mp3` 音乐，自动为其生成高质量、可直接游玩的 **Quaver 4K**（四键下落式）音游谱面。

与传统的基于简单音频阈值的生成器不同，本项目致力于模仿人类“谱师”的作谱逻辑。它不仅能精准踩中鼓点，还能理解人声和旋律的起伏，并生成极具表现力、难度可自由调节的按键排列。

### ✨ 效果与特色
*   **极致的节奏精准度**：在底层时间轴上采用了 1/48 拍的微观乐理网格，结合动态前瞻算法，节奏踩点如同钢钉般精准，不会产生微小的毫秒级偏移。
*   **动态难度缩放**：可通过输入目标难度参数（如 2.5 或 7.0），自由控制生成的谱面难度。低难度谱面干净利落，高难度谱面连打密布且极具爆发力。
*   **类人的按键排列**：第二阶段使用了生成对抗网络 (GAN) 来学习人类写谱的审美，彻底消除了反人类的鬼畜按键，按键流（Stream）和多押（Chord）极其自然。

### 🚀 快速上手 (仅使用生成功能)
如果你只想给自己的音乐生成谱面，而不需要重新训练模型，你**只需要**下载以下几个文件即可：
1.  `stage2_infer.py` (执行的主程序)
2.  `stage2_model.py` 与 `stage2_config.py`
3.  `init_model.py` 与 `init_config.py`
4.  `requirements.txt`
5.  **模型权重文件**: `best_rhythm_model.pth` 与 `best_stage2_generator.pth`（请前往项目的 Releases 页面下载）。

**使用步骤：**
1. 使用 `requirements.txt` 一键安装所需环境：
   ```bash
   pip install -r requirements.txt
   ```
   *（或者手动安装：`pip install torch numpy scipy librosa pyyaml tqdm`）*
2. 将你想要生成的 `.mp3` 音乐文件（如 `song.mp3`）放入同级目录。
3. 打开 `stage2_infer.py`，拉到最底部的代码，修改测试音乐路径和目标难度：
   ```python
   TEST_MP3 = "song.mp3"
   TEST_TARGET_DIFF = 4.5  # 在这里调节你想要的难度 (例如 1.0 ~ 10.0)
   ```
4. 运行代码：
   ```bash
   python stage2_infer.py
   ```
5. 生成完毕后，在 `ai_2stage_chart` 文件夹中找到生成的 `.qua` 谱面文件，直接导入 Quaver 游戏即可游玩！

### 🧠 模型结构解析
本项目采用了巧妙的 **两阶段解耦架构 (Two-Stage Pipeline)**：
*   **Stage 1: RhythmNet (判断何时下落)**：采用 ResNet-1D + BiLSTM + 多头注意力机制 (Multi-Head Attention)。特别引入了 **FiLM 层**，让网络能根据输入的“难度值”在底层直接对音频特征进行调制，准确预测出一条一维的“节奏点概率曲线”。
*   **Stage 2: 掩码排列生成器 (判断落在哪个轨道)**：采用 U-Net 1D 架构，并使用 WGAN-GP 框架训练。它接收音频、难度，以及**第一阶段生成的节奏掩码 (Mask)** 作为输入。通过物理掩码限制和拒绝惩罚机制，强迫模型在不破坏节奏准度的前提下，专注于学习“如何排列按键才像人类谱师”。

### ⚠️ 免责声明 (非常重要)
1. **仅限交流与学习：** 本项目及开源代码纯粹为了深度学习学术研究、技术探讨及个人娱乐而开发，不代表任何商业用途。
2. **音乐版权：** 本 AI 工具处理的 `.mp3` 音频文件版权归原词曲作者、唱片公司及相关权利人所有。使用者必须确保自身拥有处理相关音频的合法权利或符合合理使用（Fair Use）的规定。本项目作者不拥有任何测试或训练音乐的版权。
3. **游戏版权：** “Quaver” 游戏名称及其相关资源均属于 Quaver 开发团队的商标与财产。本项目是非官方项目，与 Quaver 官方团队无任何隶属、合作或背书关系。
4. **训练数据：** 本项目的 AI 模型通过大量社区谱师制作的谱面学习而来。我们对产出这些数据的谱师群体表示最崇高的敬意。本 AI 旨在为初学者提供娱乐或为人类谱师提供打底草稿，绝非用于取代人类作谱的创造力。
5. **责任阻断：** 若使用者将生成的 AI 谱面擅自上传至官方游戏服务器、公共排行榜或引发任何版权纠纷、账号封禁等后果，均由使用者自行承担，本项目及代码原作者不承担任何法律责任。请负责任地使用本工具。
```

---

