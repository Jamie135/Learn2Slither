# Learn2Slither
42 - AI Snake Game


## Introduction
This project explores reinforcement learning (RL) through a classic game: Snake. Instead of a human controlling the snake, an intelligent agent learns how to play by interacting with a 10×10 board, receiving rewards or penalties based on its actions. Over time, the agent improves its strategy, aiming to survive as long as possible and grow the snake to a length of at least 10.


## Setup

### Clone project
```bash
git clone git@github.com:Jamie135/Learn2Slither.git
```

### Install python
```bash
sudo apt install -y python3 python3-venv python3-pip
```

### Install dependencies
```bash
cd Learn2Slither
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

### Training Phase

#### Train with UI (slower, visual feedback)
Watch the snake learn in real-time with Pygame rendering:
```bash
cd src
python train.py [grid_size]
```
- Default grid size: 10x10
- Training visualization enabled
- Significantly slower due to rendering overhead

#### Train without UI (faster, recommended)
Train at maximum speed without graphics:
```bash
cd src/training
python train_no_ui.py [grid_size]
```
- Default grid size: 10x10
- No visualization, pure computation
- **Recommended for faster training**

Training saves model checkpoints to `src/models/model_*.pth` and progress data to `src/models/data.json`.

### Running the Trained AI

Display the trained snake playing with Pygame:
```bash
cd src
python main.py [grid_size]
```
- Loads the latest model (`model_9999.pth` by default)
- Shows the AI playing in real-time

#### Load a specific model checkpoint:
```bash
cd src
python main.py -model model_5000.pth
```

#### Controls during AI playback:
- **1** — Decrease speed (longer intervals)
- **2** — Increase speed (shorter intervals)
- **Enter** — Restart after game over
- **Esc** — Quit

### Play Manually (Human Mode)

Test the game yourself with keyboard controls:
```bash
cd src
python main.py -player [grid_size]
```
- **Arrow keys** — Control snake direction
- **1** / **2** — Adjust game speed
- **Enter** — Restart after game over
- **Esc** — Quit
