'use strict';

/**
 * Renderer — draws the GridWorld on a <canvas>.
 *
 * Each cell is drawn as a colored square with a small gap between cells.
 * The agent is drawn as a circle.
 *
 * When showQ is true and a QLearningAgent is provided, each empty cell
 * is split into four triangular wedges (up/right/down/left), each colored
 * by that action's Q-value. Green = high, red = low, blue = near zero.
 * This shows the full Q-value landscape at a glance.
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
    this.gap = 2;
    this.qAgent = null;       // set to a QLearningAgent to enable Q overlay
    this.showQ = false;       // toggle Q-value overlay
    this.signalField = null;  // set to a SignalField to enable signal overlay
    this.showSignals = false; // toggle signal overlay
  }

  get cellSize() {
    const usableW = this.canvas.width - this.gap * (this.env.cols + 1);
    const usableH = this.canvas.height - this.gap * (this.env.rows + 1);
    return Math.floor(Math.min(usableW / this.env.cols, usableH / this.env.rows));
  }

  cellOrigin(row, col) {
    const size = this.cellSize;
    const x = this.gap + col * (size + this.gap);
    const y = this.gap + row * (size + this.gap);
    return [x, y];
  }

  /**
   * Convert a Q-value to a color.
   * Positive values → green, negative → red, zero → neutral blue.
   */
  qColor(value, minQ, maxQ) {
    // Normalize to [-1, 1] range
    const range = Math.max(Math.abs(minQ), Math.abs(maxQ), 0.01);
    const norm = Math.max(-1, Math.min(1, value / range));

    if (norm >= 0) {
      // Blend from base blue (#0f3460) toward green
      const t = norm;
      const r = Math.round(15 * (1 - t) + 46 * t);
      const g = Math.round(52 * (1 - t) + 204 * t);
      const b = Math.round(96 * (1 - t) + 113 * t);
      return `rgb(${r},${g},${b})`;
    } else {
      // Blend from base blue toward red
      const t = -norm;
      const r = Math.round(15 * (1 - t) + 231 * t);
      const g = Math.round(52 * (1 - t) + 76 * t);
      const b = Math.round(96 * (1 - t) + 60 * t);
      return `rgb(${r},${g},${b})`;
    }
  }

  draw() {
    const ctx = this.ctx;
    const size = this.cellSize;

    // If showing Q-values, compute global min/max across ALL action values
    let minQ = 0, maxQ = 0;
    if (this.showQ && this.qAgent) {
      for (let r = 0; r < this.env.rows; r++) {
        for (let c = 0; c < this.env.cols; c++) {
          if (this.env.layout[r][c] === CELL_WALL) continue;
          const qv = this.qAgent.getQ(r, c);
          for (let a = 0; a < NUM_ACTIONS; a++) {
            if (qv[a] < minQ) minQ = qv[a];
            if (qv[a] > maxQ) maxQ = qv[a];
          }
        }
      }
    }

    // Clear
    ctx.fillStyle = COLORS.grid;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw cells
    for (let r = 0; r < this.env.rows; r++) {
      for (let c = 0; c < this.env.cols; c++) {
        const [x, y] = this.cellOrigin(r, c);
        const cell = this.env.layout[r][c];

        // Q-value wedge view for empty cells when overlay is on
        if (this.showQ && this.qAgent && cell === CELL_EMPTY) {
          const qv = this.qAgent.getQ(r, c);
          const cx = x + size / 2;
          const cy = y + size / 2;

          // Four triangles: up, right, down, left
          // Each is a triangle from center to two adjacent corners
          const corners = [
            [x, y],                 // top-left
            [x + size, y],          // top-right
            [x + size, y + size],   // bottom-right
            [x, y + size],          // bottom-left
          ];

          // Wedge mapping (swapped so the colored region is on the
          // side you'd move TOWARD, which reads more intuitively):
          //   action 0 (up)    → bottom wedge (points up toward top edge)
          //   action 1 (right) → left wedge   (points right toward right edge)
          //   action 2 (down)  → top wedge    (points down toward bottom edge)
          //   action 3 (left)  → right wedge  (points left toward left edge)
          const wedgeFor = [2, 3, 0, 1]; // swap U↔D, L↔R
          for (let a = 0; a < NUM_ACTIONS; a++) {
            const w = wedgeFor[a];
            const c1 = corners[w];
            const c2 = corners[(w + 1) % 4];

            ctx.fillStyle = this.qColor(qv[a], minQ, maxQ);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(c1[0], c1[1]);
            ctx.lineTo(c2[0], c2[1]);
            ctx.closePath();
            ctx.fill();

            // Thin border between wedges for clarity
            ctx.strokeStyle = 'rgba(0,0,0,0.3)';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(c1[0], c1[1]);
            ctx.stroke();
          }

          // Show Q-value numbers in each wedge
          const fontSize = Math.floor(size * 0.16);
          ctx.font = `${fontSize}px system-ui`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          // Position labels in the swapped wedges
          const labelPositions = [
            [cx, y + size * 0.8],           // up action → bottom wedge
            [x + size * 0.2, cy],           // right action → left wedge
            [cx, y + size * 0.2],           // down action → top wedge
            [x + size * 0.8, cy],           // left action → right wedge
          ];

          for (let a = 0; a < NUM_ACTIONS; a++) {
            ctx.fillStyle = 'rgba(255,255,255,0.8)';
            ctx.fillText(qv[a].toFixed(2), labelPositions[a][0], labelPositions[a][1]);
          }
        } else {
          // Normal cell drawing (walls, goal, pit, or plain empty when Q overlay is off)
          if (cell === CELL_WALL)      ctx.fillStyle = COLORS.wall;
          else if (cell === CELL_GOAL) ctx.fillStyle = COLORS.goal;
          else if (cell === CELL_PIT)  ctx.fillStyle = COLORS.pit;
          else                         ctx.fillStyle = COLORS.empty;

          ctx.fillRect(x, y, size, size);

          // Signal overlay: tint empty cells by signal strength
          if (this.showSignals && this.signalField && cell === CELL_EMPTY) {
            const sig = this.signalField.read(r, c);
            if (sig > 0) {
              // Positive signal: green glow (goal scent)
              const alpha = Math.min(0.6, sig * 0.6);
              ctx.fillStyle = `rgba(46, 204, 113, ${alpha})`;
              ctx.fillRect(x, y, size, size);
            } else if (sig < 0) {
              // Negative signal: red glow (pit scent)
              const alpha = Math.min(0.6, Math.abs(sig) * 0.6);
              ctx.fillStyle = `rgba(231, 76, 60, ${alpha})`;
              ctx.fillRect(x, y, size, size);
            }

            // Show numeric signal value
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            ctx.font = `${Math.floor(size * 0.2)}px system-ui`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(sig.toFixed(2), x + size / 2, y + size / 2);
          }

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
    }

    // Draw agent
    const [ax, ay] = this.cellOrigin(this.env.agentRow, this.env.agentCol);
    ctx.fillStyle = COLORS.agent;
    ctx.beginPath();
    ctx.arc(ax + size / 2, ay + size / 2, size * 0.35, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#1a1a2e';
    ctx.font = `bold ${Math.floor(size * 0.3)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('A', ax + size / 2, ay + size / 2);
  }
}
