'use strict';

/**
 * Renderer — draws the GridWorld on a <canvas>.
 *
 * Each cell is drawn as a colored square with a small gap between cells.
 * The agent is drawn as a circle.
 */

const COLORS = {
  empty: '#0f3460',
  wall:  '#1a1a2e',
  goal:  '#2ecc71',
  pit:   '#e74c3c',
  agent: '#f1c40f',
  grid:  '#16213e',  // gap color (matches canvas bg)
};

class Renderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {GridWorld} env
   */
  constructor(canvas, env) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.env = env;
    this.gap = 2; // pixels between cells
  }

  /** Recalculate cell size based on canvas dimensions and grid size. */
  get cellSize() {
    const usableW = this.canvas.width - this.gap * (this.env.cols + 1);
    const usableH = this.canvas.height - this.gap * (this.env.rows + 1);
    return Math.floor(Math.min(usableW / this.env.cols, usableH / this.env.rows));
  }

  /** Convert grid [row, col] to canvas pixel [x, y] (top-left of cell). */
  cellOrigin(row, col) {
    const size = this.cellSize;
    const x = this.gap + col * (size + this.gap);
    const y = this.gap + row * (size + this.gap);
    return [x, y];
  }

  /** Draw the full grid + agent. */
  draw() {
    const ctx = this.ctx;
    const size = this.cellSize;

    // Clear
    ctx.fillStyle = COLORS.grid;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw cells
    for (let r = 0; r < this.env.rows; r++) {
      for (let c = 0; c < this.env.cols; c++) {
        const [x, y] = this.cellOrigin(r, c);
        const cell = this.env.layout[r][c];

        if (cell === CELL_WALL)      ctx.fillStyle = COLORS.wall;
        else if (cell === CELL_GOAL) ctx.fillStyle = COLORS.goal;
        else if (cell === CELL_PIT)  ctx.fillStyle = COLORS.pit;
        else                         ctx.fillStyle = COLORS.empty;

        ctx.fillRect(x, y, size, size);

        // Label goal and pit cells
        if (cell === CELL_GOAL || cell === CELL_PIT) {
          ctx.fillStyle = '#fff';
          ctx.font = `bold ${Math.floor(size * 0.35)}px system-ui`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            cell === CELL_GOAL ? 'G' : 'P',
            x + size / 2,
            y + size / 2
          );
        }
      }
    }

    // Draw agent
    const [ax, ay] = this.cellOrigin(this.env.agentRow, this.env.agentCol);
    ctx.fillStyle = COLORS.agent;
    ctx.beginPath();
    ctx.arc(ax + size / 2, ay + size / 2, size * 0.35, 0, Math.PI * 2);
    ctx.fill();

    // Agent label
    ctx.fillStyle = '#1a1a2e';
    ctx.font = `bold ${Math.floor(size * 0.3)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('A', ax + size / 2, ay + size / 2);
  }
}
