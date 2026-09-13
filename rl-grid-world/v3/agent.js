'use strict';

/**
 * RandomAgent — picks a uniformly random action every step.
 *
 * This is the simplest possible agent. It serves as a baseline:
 * any learning algorithm should do better than this.
 *
 * The agent interface is:
 *   - act(state) → action    (choose what to do)
 *   - learn(state, action, reward, nextState, done)  (update internal state)
 *
 * RandomAgent.learn() is a no-op — it doesn't learn. But we define the
 * interface now so that Q-learning can drop in later with the same API.
 */

class RandomAgent {
  constructor() {
    this.name = 'Random';
  }

  act(_state) {
    return Math.floor(Math.random() * NUM_ACTIONS);
  }

  learn(_state, _action, _reward, _nextState, _done) {}
}

/**
 * QLearningAgent — tabular Q-learning with epsilon-greedy exploration.
 *
 * The Q-table maps each (row, col) state to an array of 4 action values.
 * All values start at 0.
 *
 * On each step the agent updates one entry using the Bellman equation:
 *
 *   Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
 *                                      ↑                    ↑
 *                             "target" (what we think       "current estimate"
 *                              the true value is)
 *
 * Hyperparameters:
 *   α (alpha)   — learning rate.  How fast we update. 0 = never learn, 1 = overwrite.
 *   γ (gamma)   — discount factor. How much we value future reward. 0 = myopic, 1 = patient.
 *   ε (epsilon) — exploration rate. Probability of picking a random action instead of the best.
 */

class QLearningAgent {
  constructor(rows, cols, { alpha = 0.1, gamma = 0.95, epsilon = 0.1 } = {}) {
    this.name = 'Q-Learning';
    this.rows = rows;
    this.cols = cols;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;

    // Q-table: q[row][col][action] = value.  Initialized to 0.
    this.q = [];
    for (let r = 0; r < rows; r++) {
      this.q[r] = [];
      for (let c = 0; c < cols; c++) {
        this.q[r][c] = new Float64Array(NUM_ACTIONS); // [0, 0, 0, 0]
      }
    }

    this.totalUpdates = 0;
  }

  /** Epsilon-greedy action selection. */
  act(state) {
    const [r, c] = state;

    // With probability epsilon, explore (random action)
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * NUM_ACTIONS);
    }

    // Otherwise, exploit: pick the action with highest Q-value.
    // Break ties randomly.
    const qValues = this.q[r][c];
    let bestVal = qValues[0];
    let bestActions = [0];

    for (let a = 1; a < NUM_ACTIONS; a++) {
      if (qValues[a] > bestVal) {
        bestVal = qValues[a];
        bestActions = [a];
      } else if (qValues[a] === bestVal) {
        bestActions.push(a);
      }
    }

    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }

  /**
   * Q-learning update (off-policy: uses max over next actions).
   *
   * The key insight: we don't need to know the environment's rules.
   * We just observe what happened (s, a, r, s') and update our estimate.
   */
  learn(state, action, reward, nextState, done) {
    const [r, c] = state;
    const [nr, nc] = nextState;

    const currentQ = this.q[r][c][action];

    // If the episode ended, there's no future reward
    let target;
    if (done) {
      target = reward;
    } else {
      // Future value = best we can do from the next state
      const nextQ = this.q[nr][nc];
      const maxNextQ = Math.max(nextQ[0], nextQ[1], nextQ[2], nextQ[3]);
      target = reward + this.gamma * maxNextQ;
    }

    // Update: nudge Q(s,a) toward the target
    // The (target - currentQ) is the "TD error" — how wrong we were
    this.q[r][c][action] = currentQ + this.alpha * (target - currentQ);
    this.totalUpdates++;
  }

  /** Get the Q-values for a specific cell (for visualization). */
  getQ(row, col) {
    return Array.from(this.q[row][col]);
  }

  /** Get the best action's value for a cell (for heatmap). */
  getMaxQ(row, col) {
    const qv = this.q[row][col];
    return Math.max(qv[0], qv[1], qv[2], qv[3]);
  }

  /** Reset Q-table to zeros. */
  resetQ() {
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        this.q[r][c].fill(0);
      }
    }
    this.totalUpdates = 0;
  }
}
