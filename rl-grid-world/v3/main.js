'use strict';

/**
 * Main — wires up manual control, agent selection, Q-learning, and stats.
 */

// --- Setup ---
const env = new GridWorld(DEFAULT_LAYOUT, DEFAULT_START);
const canvas = document.getElementById('grid');
const renderer = new Renderer(canvas, env);
const statusEl = document.getElementById('status');

// --- Agents ---
const randomAgent = new RandomAgent();
const qAgent = new QLearningAgent(env.rows, env.cols, {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 0.1
});
renderer.qAgent = qAgent;

// --- Signal field ---
const signalField = new SignalField(env);
renderer.signalField = signalField;

function getActiveAgent() {
  const sel = document.getElementById('agentSelect').value;
  return sel === 'qlearn' ? qAgent : randomAgent;
}

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

  const cell = env.layout[env.agentRow][env.agentCol];
  if (cell === CELL_GOAL) stats.goals++;
  else if (cell === CELL_PIT) stats.pits++;
  else stats.timeouts++;

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

// --- Sync Q-learning hyperparameters from UI ---
function syncQParams() {
  qAgent.alpha   = Number(document.getElementById('alpha').value)   || 0.1;
  qAgent.gamma   = Number(document.getElementById('gamma').value)   || 0.95;
  qAgent.epsilon = Number(document.getElementById('epsilon').value) || 0.1;
}

// --- Animation state ---
let running = false;

function setButtonsEnabled(enabled) {
  document.getElementById('runOne').disabled = !enabled;
  document.getElementById('runBatch').disabled = !enabled;
  document.getElementById('resetStats').disabled = !enabled;
  document.getElementById('resetQ').disabled = !enabled;
}

function getStepDelay() {
  const speed = Number(document.getElementById('speed').value);
  return Math.max(10, Math.round(1000 / speed));
}

// --- Run one visible episode ---
async function runOneEpisode() {
  if (running) return;
  running = true;
  setButtonsEnabled(false);
  syncQParams();

  const agent = getActiveAgent();
  env.reset();
  renderer.draw();

  while (!env.done) {
    const state = env.getState();
    const action = agent.act(state);
    const result = env.step(action);

    // Learn from this transition (no-op for RandomAgent)
    agent.learn(state, action, result.reward, result.state, result.done);

    renderer.draw();
    statusEl.textContent = `${agent.name} | Step ${env.steps} | Reward: ${env.totalReward.toFixed(2)}`;

    await sleep(getStepDelay());
  }

  recordEpisode(env);
  showEndStatus(agent.name);

  running = false;
  setButtonsEnabled(true);
}

// --- Run a batch of episodes silently ---
function runBatch(n) {
  if (running) return;
  syncQParams();
  const agent = getActiveAgent();

  for (let i = 0; i < n; i++) {
    env.reset();
    while (!env.done) {
      const state = env.getState();
      const action = agent.act(state);
      const result = env.step(action);
      agent.learn(state, action, result.reward, result.state, result.done);
    }
    recordEpisode(env);
  }

  renderer.draw();
  statusEl.textContent = `${agent.name}: ran ${n} episodes. See stats panel.`;
}

function showEndStatus(agentName) {
  const cell = env.layout[env.agentRow][env.agentCol];
  if (cell === CELL_GOAL) {
    statusEl.textContent = `${agentName} reached goal! ${env.steps} steps, reward ${env.totalReward.toFixed(2)}`;
  } else if (cell === CELL_PIT) {
    statusEl.textContent = `${agentName} fell in pit. ${env.steps} steps, reward ${env.totalReward.toFixed(2)}`;
  } else {
    statusEl.textContent = `${agentName} timed out. ${env.steps} steps, reward ${env.totalReward.toFixed(2)}`;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Initial draw ---
env.reset();
renderer.draw();
updateStatsDisplay();

// --- Keyboard: manual control ---
document.addEventListener('keydown', (e) => {
  if (running) return;

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
      statusEl.textContent = 'Reset. Use arrow keys or run an agent.';
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

document.getElementById('resetQ').addEventListener('click', () => {
  qAgent.resetQ();
  resetStats();
  env.reset();
  renderer.draw();
  statusEl.textContent = 'Q-table and stats reset.';
});

document.getElementById('speed').addEventListener('input', () => {
  const speed = document.getElementById('speed').value;
  document.getElementById('speedLabel').textContent = `${speed} steps/sec`;
});

document.getElementById('showQ').addEventListener('change', (e) => {
  renderer.showQ = e.target.checked;
  if (e.target.checked) {
    // Turn off signals overlay when Q is on (they overlap)
    renderer.showSignals = false;
    document.getElementById('showSignals').checked = false;
  }
  renderer.draw();
});

document.getElementById('showSignals').addEventListener('change', (e) => {
  renderer.showSignals = e.target.checked;
  if (e.target.checked) {
    // Turn off Q overlay when signals is on
    renderer.showQ = false;
    document.getElementById('showQ').checked = false;
  }
  renderer.draw();
});

// Live-update hyperparameters
for (const id of ['alpha', 'gamma', 'epsilon']) {
  document.getElementById(id).addEventListener('change', syncQParams);
}
