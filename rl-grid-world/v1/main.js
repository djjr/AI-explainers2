'use strict';

/**
 * Main — wires up manual control, random agent, and episode statistics.
 *
 * You can still drive the agent with arrow keys. But now you can also
 * let the RandomAgent play — either one visible episode or a batch of
 * 100 silent episodes to collect statistics.
 */

// --- Setup ---
const env = new GridWorld(DEFAULT_LAYOUT, DEFAULT_START);
const canvas = document.getElementById('grid');
const renderer = new Renderer(canvas, env);
const statusEl = document.getElementById('status');
const agent = new RandomAgent();

// --- Statistics tracking ---
const stats = {
  episodes: 0,
  totalSteps: 0,
  totalReward: 0,
  goals: 0,
  pits: 0,
  timeouts: 0,
};

function resetStats() {
  stats.episodes = 0;
  stats.totalSteps = 0;
  stats.totalReward = 0;
  stats.goals = 0;
  stats.pits = 0;
  stats.timeouts = 0;
  updateStatsDisplay();
}

function recordEpisode(env) {
  stats.episodes++;
  stats.totalSteps += env.steps;
  stats.totalReward += env.totalReward;

  // Figure out how the episode ended
  const cell = env.layout[env.agentRow][env.agentCol];
  if (cell === CELL_GOAL) stats.goals++;
  else if (cell === CELL_PIT) stats.pits++;
  else stats.timeouts++; // hit maxSteps without reaching goal or pit

  updateStatsDisplay();
}

function updateStatsDisplay() {
  const el = (id) => document.getElementById(id);
  el('statEpisodes').textContent = stats.episodes;
  el('statGoals').textContent = stats.goals;
  el('statPits').textContent = stats.pits;
  el('statTimeouts').textContent = stats.timeouts;

  if (stats.episodes > 0) {
    el('statAvgSteps').textContent = (stats.totalSteps / stats.episodes).toFixed(1);
    el('statAvgReward').textContent = (stats.totalReward / stats.episodes).toFixed(2);
    el('statSuccessRate').textContent = ((stats.goals / stats.episodes) * 100).toFixed(1) + '%';
  } else {
    el('statAvgSteps').textContent = '—';
    el('statAvgReward').textContent = '—';
    el('statSuccessRate').textContent = '—';
  }
}

// --- Animation state ---
let running = false;  // true while an animated episode is playing

function setButtonsEnabled(enabled) {
  document.getElementById('runOne').disabled = !enabled;
  document.getElementById('runBatch').disabled = !enabled;
  document.getElementById('resetStats').disabled = !enabled;
}

function getStepDelay() {
  const speed = Number(document.getElementById('speed').value);
  // speed 1 = 1000ms, speed 100 = 10ms
  return Math.max(10, Math.round(1000 / speed));
}

// --- Run one visible episode ---
async function runOneEpisode() {
  if (running) return;
  running = true;
  setButtonsEnabled(false);

  env.reset();
  renderer.draw();

  while (!env.done) {
    const state = env.getState();
    const action = agent.act(state);
    const result = env.step(action);
    renderer.draw();

    // Update status during animation
    statusEl.textContent = `Random agent | Step ${env.steps} | Reward: ${env.totalReward.toFixed(2)}`;

    // Wait so the user can see each step
    await sleep(getStepDelay());
  }

  recordEpisode(env);

  // Final status
  const cell = env.layout[env.agentRow][env.agentCol];
  if (cell === CELL_GOAL) {
    statusEl.textContent = `Random agent reached goal! ${env.steps} steps, reward ${env.totalReward.toFixed(2)}`;
  } else if (cell === CELL_PIT) {
    statusEl.textContent = `Random agent fell in pit. ${env.steps} steps, reward ${env.totalReward.toFixed(2)}`;
  } else {
    statusEl.textContent = `Random agent timed out. ${env.steps} steps, reward ${env.totalReward.toFixed(2)}`;
  }

  running = false;
  setButtonsEnabled(true);
}

// --- Run a batch of episodes silently ---
function runBatch(n) {
  if (running) return;

  for (let i = 0; i < n; i++) {
    env.reset();
    while (!env.done) {
      const state = env.getState();
      const action = agent.act(state);
      env.step(action);
    }
    recordEpisode(env);
  }

  // Show final state of last episode
  renderer.draw();
  statusEl.textContent = `Ran ${n} episodes. See stats panel.`;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Initial draw ---
env.reset();
renderer.draw();
updateStatsDisplay();

// --- Keyboard: manual control (same as v0) ---
document.addEventListener('keydown', (e) => {
  if (running) return; // don't interfere with animation

  let action = null;
  switch (e.key) {
    case 'ArrowUp':    action = ACTION_UP;    break;
    case 'ArrowRight': action = ACTION_RIGHT; break;
    case 'ArrowDown':  action = ACTION_DOWN;  break;
    case 'ArrowLeft':  action = ACTION_LEFT;  break;
    case 'r':
    case 'R':
      env.reset();
      renderer.draw();
      statusEl.textContent = 'Reset. Use arrow keys or run the random agent.';
      return;
  }

  if (action === null) return;
  e.preventDefault();

  const result = env.step(action);
  renderer.draw();

  if (result.done && result.reward > 0) {
    statusEl.textContent = `Goal! Reward: ${env.totalReward.toFixed(2)} in ${env.steps} steps. Press R to reset.`;
  } else if (result.done && result.reward < 0) {
    statusEl.textContent = `Pit! Reward: ${env.totalReward.toFixed(2)} in ${env.steps} steps. Press R to reset.`;
  } else {
    statusEl.textContent = `Moved ${ACTION_NAMES[action]} | Step ${env.steps} | Reward: ${env.totalReward.toFixed(2)}`;
  }
});

// --- Button handlers ---
document.getElementById('runOne').addEventListener('click', runOneEpisode);
document.getElementById('runBatch').addEventListener('click', () => runBatch(100));
document.getElementById('resetStats').addEventListener('click', () => {
  resetStats();
  env.reset();
  renderer.draw();
  statusEl.textContent = 'Stats reset.';
});

document.getElementById('speed').addEventListener('input', () => {
  const speed = document.getElementById('speed').value;
  document.getElementById('speedLabel').textContent = `${speed} steps/sec`;
});
