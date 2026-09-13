'use strict';

/**
 * GridWorld — a simple grid environment.
 *
 * Concepts:
 *   - The world is a ROWS x COLS grid of cells.
 *   - Each cell can be: empty, wall, goal, or pit.
 *   - The agent occupies one cell and can move in 4 directions.
 *   - Moving into a wall or off the edge does nothing (agent stays put).
 *   - Reaching a goal cell ends the episode with reward +1.
 *   - Reaching a pit cell ends the episode with reward -1.
 *   - Each step that isn't terminal gives reward -0.01 (small cost of living
 *     to encourage finding the goal efficiently).
 *
 * Actions: 0=up, 1=right, 2=down, 3=left
 */

const CELL_EMPTY = 0;
const CELL_WALL  = 1;
const CELL_GOAL  = 2;
const CELL_PIT   = 3;

const ACTION_UP    = 0;
const ACTION_RIGHT = 1;
const ACTION_DOWN  = 2;
const ACTION_LEFT  = 3;
const ACTION_NAMES = ['up', 'right', 'down', 'left'];
const NUM_ACTIONS  = 4;

// Direction deltas: [dRow, dCol] for each action
const DELTAS = [
  [-1,  0], // up
  [ 0,  1], // right
  [ 1,  0], // down
  [ 0, -1], // left
];

class GridWorld {
  /**
   * @param {number[][]} layout - 2D array of cell types (CELL_* constants)
   * @param {[number,number]} startPos - [row, col] starting position
   */
  constructor(layout, startPos) {
    this.layout = layout;
    this.rows = layout.length;
    this.cols = layout[0].length;
    this.startPos = startPos;

    // Current state
    this.agentRow = startPos[0];
    this.agentCol = startPos[1];
    this.done = false;
    this.totalReward = 0;
    this.steps = 0;
  }

  /** Reset the environment to starting state. Returns the initial state. */
  reset() {
    this.agentRow = this.startPos[0];
    this.agentCol = this.startPos[1];
    this.done = false;
    this.totalReward = 0;
    this.steps = 0;
    return this.getState();
  }

  /** Returns the current state as [row, col]. */
  getState() {
    return [this.agentRow, this.agentCol];
  }

  /**
   * Take an action. Returns { state, reward, done }.
   * @param {number} action - one of ACTION_UP/RIGHT/DOWN/LEFT
   */
  step(action) {
    if (this.done) {
      return { state: this.getState(), reward: 0, done: true };
    }

    const [dr, dc] = DELTAS[action];
    const newRow = this.agentRow + dr;
    const newCol = this.agentCol + dc;

    // Boundary check
    if (newRow < 0 || newRow >= this.rows || newCol < 0 || newCol >= this.cols) {
      // Stayed in place — still costs a step
      this.steps++;
      this.totalReward += -0.01;
      return { state: this.getState(), reward: -0.01, done: false };
    }

    // Wall check
    if (this.layout[newRow][newCol] === CELL_WALL) {
      this.steps++;
      this.totalReward += -0.01;
      return { state: this.getState(), reward: -0.01, done: false };
    }

    // Move the agent
    this.agentRow = newRow;
    this.agentCol = newCol;
    this.steps++;

    // Check what we landed on
    const cell = this.layout[newRow][newCol];
    let reward = -0.01;

    if (cell === CELL_GOAL) {
      reward = 1.0;
      this.done = true;
    } else if (cell === CELL_PIT) {
      reward = -1.0;
      this.done = true;
    }

    this.totalReward += reward;
    return { state: this.getState(), reward, done: this.done };
  }
}

// --- Default layout: a simple 6x6 grid ---
//
//  . . . . . G
//  . W W . . .
//  . . . . W .
//  . . P . . .
//  . W . . W .
//  A . . . . .
//
// A = agent start (5,0), G = goal (0,5), P = pit (3,2), W = walls

const DEFAULT_LAYOUT = [
  [0, 0, 0, 0, 0, 2],  // row 0: goal at (0,5)
  [0, 1, 1, 0, 0, 0],  // row 1: walls
  [0, 0, 0, 0, 1, 0],  // row 2: wall
  [0, 0, 3, 0, 0, 0],  // row 3: pit at (3,2)
  [0, 1, 0, 0, 1, 0],  // row 4: walls
  [0, 0, 0, 0, 0, 0],  // row 5: agent starts at (5,0)
];

const DEFAULT_START = [5, 0];
