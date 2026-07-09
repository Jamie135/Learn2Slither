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

```bash
cd src/
```

### Training Phase

Train the AI agent using reinforcement learning. By default, training runs headless (no graphics) for maximum speed.

```bash
python main.py [grid_size] [-sessions N] [-save PATH] [-load PATH] [-visual]
```

**Arguments:**
- `grid_size` — Board size (default: 10, minimum: 4)
- `-sessions` — Number of training sessions (default: 1)
- `-save` — Path to save the trained model (e.g., models/100sess.pth)
- `-load` — Path to load a previously trained model to continue training
- `-visual` — Enable graphical UI (disabled by default)
- `-state` — Enable terminal display of the agent's state

**Examples:**

Train 10 sessions and save the model:
```bash
python main.py -sessions 10 -save models/10sess.pth
```

Train 100 sessions, continuing from a saved model:
```bash
python main.py -sessions 100 -load models/10sess.pth -save models/100sess.pth
```

Train with visualization (slower, but watch the AI learn):
```bash
python main.py -sessions 100 -save models/100sess.pth -visual
```

Train on a different grid size (e.g., 6×6):
```bash
python main.py 6 -sessions 100 -save models/6x6_100sess.pth
```

**Alternative:** You can also use `train.py` for training-only workflows:
```bash
python train.py -sessions 10000 -save models/10000sess.pth -visual
```

### Evaluating Trained Models

Run a trained model in evaluation mode (no learning) to test its performance:

```bash
python main.py [grid_size] [-load PATH] [-sessions N] [-eval] [-visual] [-step]
```

**Arguments:**
- `-load` — Path to the trained model file (required)
- `-sessions` — Number of evaluation sessions to run (default: 1)
- `-eval` — Disable learning (evaluation mode)
- `-visual` — Show graphical UI (default: on)
- `-state` — Enable terminal display of the snake's state
- `-step` — Step through each move with keypress (Space/Enter/arrows)

**Examples:**

Evaluate a model visually over 10 sessions:
```bash
python main.py -load models/100sess.pth -sessions 10 -eval -visual
```

Step-by-step evaluation (press Space/Enter to advance each move) with the agent's state displayed on terminal:
```bash
python main.py -load models/100sess.pth -sessions 1 -eval -state -step
```

Headless evaluation (fast performance testing):
```bash
python main.py -load models/100sess.pth -sessions 100 -eval -visual
```

#### Controls during AI playback:
- **1** — Decrease speed (longer intervals)
- **2** — Increase speed (shorter intervals)
- **Space/Enter/Arrows** — Advance one step (in step mode)
- **Esc** — Quit

### Play Manually (Human Mode)

Test the game yourself with keyboard controls:

```bash
python main.py [grid_size] -player
```

**Arguments:**
- `grid_size` — Board size (default: 10, minimum: 4)
- `-player` — Enable gameplay mode for human control

**Example:**
```bash
python main.py 10 -player
```

#### Controls during gameplay:
- **Arrow keys** — Control snake direction
- **1** / **2** — Adjust game speed
- **Enter** — Start game / Restart after game over
- **Esc** — Quit
