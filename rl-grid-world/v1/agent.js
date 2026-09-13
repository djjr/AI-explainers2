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

  /** Pick a random action (ignores state entirely). */
  act(_state) {
    return Math.floor(Math.random() * NUM_ACTIONS);
  }

  /** No-op — random agent doesn't learn. */
  learn(_state, _action, _reward, _nextState, _done) {}
}
