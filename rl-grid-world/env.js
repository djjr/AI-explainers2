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
 *   - Episodes end after maxSteps to prevent infinite wandering.
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
  constructor(layout, startPos, maxSteps = 200) {
    this.layout = layout;
    this.rows = layout.length;
    this.cols = layout[0].length;
    this.startPos = startPos;
    this.maxSteps = maxSteps;

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

    // Timeout check
    if (!this.done && this.steps >= this.maxSteps) {
      this.done = true;
    }

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

/**
 * SignalField — computes a "scent" field over the grid.
 *
 * Each emitter (goal, pit, or agent) radiates a signal that decays
 * with Manhattan distance. Walls block signal propagation (we use
 * a BFS flood-fill to compute path distance, not straight-line distance,
 * so signals curve around walls).
 *
 * The field is a 2D array of numbers: positive near goals, negative
 * near pits. An agent at cell (r,c) can read the signal value and
 * the signal gradient (difference between neighboring cells) to
 * decide which way to go.
 *
 * decay formula:  signal(d) = strength / (1 + d)
 *   where d = shortest walkable path distance (BFS) from the emitter.
 */

class SignalField {
  constructor(env) {
    this.env = env;
    this.rows = env.rows;
    this.cols = env.cols;

    // The combined signal at each cell
    this.field = this._createGrid(0);

    // Precompute emitter locations from the layout
    this.emitters = [];
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (env.layout[r][c] === CELL_GOAL) {
          this.emitters.push({ row: r, col: c, strength: 1.0 });
        } else if (env.layout[r][c] === CELL_PIT) {
          this.emitters.push({ row: r, col: c, strength: -1.0 });
        }
      }
    }

    this.compute();
  }

  _createGrid(val) {
    const g = [];
    for (let r = 0; r < this.rows; r++) {
      g[r] = new Float64Array(this.cols);
      if (val !== 0) g[r].fill(val);
    }
    return g;
  }

  /**
   * BFS from a single cell, returning path distances.
   * Walls are impassable. Returns a grid of distances (-1 = unreachable).
   */
  _bfsDistances(startRow, startCol) {
    const dist = this._createGrid(-1);
    dist[startRow][startCol] = 0;
    const queue = [[startRow, startCol]];
    let head = 0;

    while (head < queue.length) {
      const [r, c] = queue[head++];
      const d = dist[r][c];

      for (const [dr, dc] of DELTAS) {
        const nr = r + dr;
        const nc = c + dc;
        if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) continue;
        if (this.env.layout[nr][nc] === CELL_WALL) continue;
        if (dist[nr][nc] >= 0) continue; // already visited
        dist[nr][nc] = d + 1;
        queue.push([nr, nc]);
      }
    }

    return dist;
  }

  /** Recompute the full signal field from all emitters. */
  compute() {
    // Zero out
    for (let r = 0; r < this.rows; r++) {
      this.field[r].fill(0);
    }

    // Accumulate signal from each emitter
    for (const em of this.emitters) {
      const dist = this._bfsDistances(em.row, em.col);
      for (let r = 0; r < this.rows; r++) {
        for (let c = 0; c < this.cols; c++) {
          if (dist[r][c] < 0) continue; // unreachable
          this.field[r][c] += em.strength / (1 + dist[r][c]);
        }
      }
    }
  }

  /** Read the combined signal at a cell. */
  read(row, col) {
    return this.field[row][col];
  }

  /**
   * Read the signal gradient: the signal value in each of the 4 neighboring
   * cells (or the current cell if the neighbor is a wall/boundary).
   * Returns [up, right, down, left] signal values.
   */
  gradient(row, col) {
    const result = [];
    for (const [dr, dc] of DELTAS) {
      const nr = row + dr;
      const nc = col + dc;
      if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) {
        result.push(this.field[row][col]); // boundary: same as current
      } else if (this.env.layout[nr][nc] === CELL_WALL) {
        result.push(this.field[row][col]); // wall: same as current
      } else {
        result.push(this.field[nr][nc]);
      }
    }
    return result; // [up, right, down, left]
  }
}

const DEFAULT_LAYOUT = [
  [0, 0, 0, 0, 0, 2],  // row 0: goal at (0,5)
  [0, 1, 1, 0, 0, 0],  // row 1: walls
  [0, 0, 0, 0, 1, 0],  // row 2: wall
  [0, 0, 3, 0, 0, 0],  // row 3: pit at (3,2)
  [0, 1, 0, 0, 1, 0],  // row 4: walls
  [0, 0, 0, 0, 0, 0],  // row 5: agent starts at (5,0)
];

const DEFAULT_START = [5, 0];

// --- Transfer test layout: different shape, same idea ---
//
//  . . . . . .
//  . . . W . .
//  G . . W . .
//  . . . . . P
//  . W W . . .
//  . . . . . A
//
// A = agent start (5,5), G = goal (2,0), P = pit (3,5)

const TRANSFER_LAYOUT = [
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [2, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 3],
  [0, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
];

const TRANSFER_START = [5, 5];
