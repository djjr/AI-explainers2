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

/**
 * SignalQAgent — Q-learning where the state is what the agent SENSES,
 * not where it IS.
 *
 * Instead of Q[row][col][action], we have Q[signalState][action].
 *
 * The signal state is built from the gradient of the signal field:
 * for each of the 4 directions, we compute
 *   delta = signal(neighbor) - signal(here)
 * and discretize it into one of 3 bins:
 *   0 = negative (signal gets worse that way)
 *   1 = ~zero    (signal stays about the same)
 *   2 = positive (signal gets better that way)
 *
 * This gives 3^4 = 81 possible states. The agent learns rules like
 * "when signal improves to the right, go right" — which transfer
 * to any grid with the same type of signals.
 *
 * The threshold parameter controls how big a delta must be to count
 * as positive or negative (vs. ~zero).
 */

class SignalQAgent {
  constructor(signalField, { alpha = 0.1, gamma = 0.95, epsilon = 0.1, threshold = 0.05 } = {}) {
    this.name = 'Signal Q';
    this.signalField = signalField;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.threshold = threshold;

    // 81 possible signal states, 4 actions each
    this.numStates = 81; // 3^4
    this.q = [];
    for (let s = 0; s < this.numStates; s++) {
      this.q[s] = new Float64Array(NUM_ACTIONS);
    }

    this.totalUpdates = 0;
  }

  /**
   * Convert a grid position to a signal-based state index.
   *
   * Reads the signal gradient (how signal changes in each direction)
   * and discretizes into bins:
   *   delta < -threshold  →  0 (worse)
   *   |delta| <= threshold →  1 (neutral)
   *   delta > threshold   →  2 (better)
   *
   * Encodes as a base-3 number: up*27 + right*9 + down*3 + left*1
   */
  _stateIndex(gridState) {
    const [row, col] = gridState;
    const here = this.signalField.read(row, col);
    const neighbors = this.signalField.gradient(row, col); // [up, right, down, left]

    let index = 0;
    const multipliers = [27, 9, 3, 1]; // base-3 place values

    for (let d = 0; d < 4; d++) {
      const delta = neighbors[d] - here;
      let bin;
      if (delta < -this.threshold)      bin = 0; // worse
      else if (delta > this.threshold)  bin = 2; // better
      else                              bin = 1; // neutral
      index += bin * multipliers[d];
    }

    return index;
  }

  act(gridState) {
    const s = this._stateIndex(gridState);

    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * NUM_ACTIONS);
    }

    const qValues = this.q[s];
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

  learn(gridState, action, reward, nextGridState, done) {
    const s = this._stateIndex(gridState);
    const ns = this._stateIndex(nextGridState);

    const currentQ = this.q[s][action];

    let target;
    if (done) {
      target = reward;
    } else {
      const nextQ = this.q[ns];
      const maxNextQ = Math.max(nextQ[0], nextQ[1], nextQ[2], nextQ[3]);
      target = reward + this.gamma * maxNextQ;
    }

    this.q[s][action] = currentQ + this.alpha * (target - currentQ);
    this.totalUpdates++;
  }

  resetQ() {
    for (let s = 0; s < this.numStates; s++) {
      this.q[s].fill(0);
    }
    this.totalUpdates = 0;
  }
}
