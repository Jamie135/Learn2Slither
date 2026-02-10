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

Train the AI agent using reinforcement learning. By default, training runs headless (no graphics) for maximum speed.

```bash
python train.py [grid_size] [-ep episodes] [-ui]
```

**Arguments:**
- `grid_size` — Board size (default: 10, minimum: 4)
- `-ep` — Number of training episodes (default: 10000)
- `-ui` — Enable graphical UI to watch training in real-time (slower)

**Examples:**

By default, fast headless training (recommended):
```bash
python train.py
```

Training with visualization (slower, but watch the AI learn):
```bash
python train.py -ui
```

Training with other grid size (e.g. 6x6):
```bash
python train.py 6
```

Training with other number of episodes (e.g. 100):
```bash
python train.py -ep 100
```

Training saves model checkpoints to `src/models/model_*.pth` and progress data to `src/models/data.json`.

### Running the Trained AI

Display the trained snake playing with Pygame:
```bash
python main.py [grid_size] [-model model.pth]
```
**Arguments**
- `grid_size` — Board size (default: 10, minimum: 4)
- `-model` Loads the model (`model_9999.pth` by default)

#### Load a specific model checkpoint:
```bash
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
python main.py [grid_size] [-player]
```
**Arguments**
- `grid_size` — Board size (default: 10, minimum: 4)
- `-player` — Enable gameplay mode for user

#### Controls during gameplay:
- **Arrow keys** — Control snake direction
- **1** / **2** — Adjust game speed
- **Enter** — Restart after game over
- **Esc** — Quit
