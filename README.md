# RL Algorithm Implementation

A collection of reinforcement learning implementations built from scratch, covering both a classic tabular control problem and an LLM-agent-style negotiation environment built with LangGraph.

## Projects

| Project | Algorithm | Environment |
|---|---|---|
| [`simple_rl_implementation/`](#1-simple_rl_implementation--taxi-sarsa) | SARSA (on-policy TD control) | `Taxi-v4` (Gymnasium) |
| [`rl_with_langgraph/`](#2-rl_with_langgraph--negotiation-agent) | Q-Learning | Custom price-negotiation game orchestrated as a LangGraph state machine |

---

## 1. `simple_rl_implementation/` — Taxi SARSA

A tabular **SARSA** agent trained to solve the `Taxi-v4` environment from [Gymnasium](https://gymnasium.farama.org/), implemented from scratch with no RL libraries.

### Overview

The classic Taxi problem: a taxi navigates a 5x5 grid to pick up a passenger at one of four locations (R, G, Y, B) and drop them off at another. This project tracks the agent's learning progress across training, evaluation, and testing.

The three stages run end-to-end from `sarsa_taxi.py`:

1. **Training** — learns a Q-table over 5000 episodes using the SARSA update rule.
2. **Evaluation** — replays 5 episodes with the trained (greedy) policy, rendered live with a custom OpenCV top-down visualization, and plots reward/episode-length curves.
3. **Testing** — runs 10 greedy-policy episodes with no rendering and reports average reward, average episode length, and success rate.

### Structure

```
simple_rl_implementation/
├── sarsa_taxi.py              # Entry point — runs training, evaluation, and testing
├── sarsa_training.py          # SARSA training loop and Q-table learning
├── sarsa_evaluation.py        # Greedy-policy evaluation + OpenCV visualization
├── sarsa_testing.py           # Greedy-policy testing and success-rate reporting
├── epsilon_greedy_policy.py   # Shared epsilon-greedy action selection
└── sarsa_q_table.pkl          # Saved Q-table from a completed training run
```

### Algorithm

The agent uses the standard SARSA (State-Action-Reward-State-Action) update:

```
Q(s, a) ← Q(s, a) + α [ r + γ · Q(s', a') − Q(s, a) ]
```

where `a'` is the action actually chosen by the current epsilon-greedy policy in the next state — making SARSA an on-policy method (unlike Q-learning, which bootstraps off the max action).

### Usage

```
cd simple_rl_implementation
python sarsa_taxi.py
```

This will:

- Train a fresh Q-table and save it to `sarsa_q_table.pkl`
- Display training reward/episode-length plots
- Open an OpenCV window visualizing 5 evaluation episodes using the trained policy
- Print testing metrics (average reward, average episode length, success rate) over 10 episodes and display the corresponding plots

To reuse an existing Q-table instead of retraining, comment out the training step in `sarsa_taxi.py` and point `q_table` to `sarsa_q_table.pkl`.

Training, evaluation, and testing plots, along with a recorded demo of the evaluation visualization (`training ui.mp4`), are in [`Results(Simple implementation)/`](<Results(Simple implementation)>).

---

## 2. `rl_with_langgraph/` — Negotiation Agent

A **Q-learning** agent that learns to negotiate a price with a simulated customer, with the multi-round negotiation loop modeled as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph rather than a plain Python loop.

### Overview

The agent starts at a fixed asking price (100) and negotiates against a customer who counters using one of three strategies:

- **patient** — concedes slowly (target rises 3/round)
- **impatient** — concedes quickly (target rises 8/round)
- **tit_for_tat** — mirrors half the agent's last concession

Each round, the agent chooses one of five actions — `hold`, `small_concede`, `big_concede`, `accept`, `walk_away` — and the customer responds with a counter-offer. A deal closes automatically once the agent's ask drops to or below the customer's offer, or the negotiation ends in a walk-away or a 6-round timeout.

### Structure

```
rl_with_langgraph/
├── main.py               # Entry point — trains, then tests, the agent
├── QLearning_agent.py     # Tabular Q-learning agent (state = round + offer-gap bucket)
├── customer.py            # Customer counter-offer strategies (patient / impatient / tit_for_tat)
├── negotiation_game.py    # LangGraph state machine wiring agent ↔ customer ↔ learning update
├── training.py            # Training loop (3000 episodes) + reward/epsilon/offer/deal-rate plots
└── testing.py              # Greedy-policy evaluation (200 episodes per customer strategy) + plots
```

### How the LangGraph loop works

`negotiation_game.py` builds a small cyclic graph with three nodes:

```
agent_move → customer_move → learn ──┬─→ agent_move (loop)
                                      └─→ END (when done)
```

- **`agent_move`** — the Q-learning agent picks an action from its current state (round number + bucketed offer gap) and updates its own offer.
- **`customer_move`** — the customer counters based on its strategy; a deal triggers automatically if the offers cross.
- **`learn`** — computes the round's reward, checks for timeout, and applies the Q-learning update before looping back or ending the episode.

### Reward structure

- **Deal**: `deal_price − cost_floor(60)` — bigger closed deals are worth more
- **Walk away**: `-5`
- **Timeout** (6 rounds, no deal): `0`
- **Any other step**: `-0.5` (small cost per round, encouraging faster deals)

### Usage

```
cd rl_with_langgraph
python -m rl_with_langgraph.main
```

This trains the agent for 3000 episodes (rotating randomly between the three customer strategies), plots reward, epsilon decay, final-offer, and deal-rate curves, then evaluates the greedy policy for 200 episodes per strategy and plots the same set of test metrics.

Training and testing plots (final offer, deal rate, epsilon decay, reward) are in [`Results (Bargain-Agent)/`](<Results (Bargain-Agent)>).

---

## Setup

```
git clone https://github.com/Madiha-06/RL_algorithm_Implementation.git
cd RL_algorithm_Implementation
pip install -r requirements.txt
```

## Requirements

Key dependencies (see `requirements.txt` for the full pinned list):

- `gymnasium` / `gym` — Taxi environment
- `numpy` — Q-table and numerical operations
- `opencv-python` — custom Taxi visualization
- `matplotlib` — training/testing plots
- `langgraph`, `langchain-core` — negotiation-agent state graph
