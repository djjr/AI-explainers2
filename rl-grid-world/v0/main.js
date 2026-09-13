'use strict';

/**
 * Main — wires up the environment, renderer, and keyboard input.
 *
 * Step 1: Manual control. You drive the agent with arrow keys.
 * This lets you get a feel for the environment before we add RL.
 */

const env = new GridWorld(DEFAULT_LAYOUT, DEFAULT_START);
const canvas = document.getElementById('grid');
const renderer = new Renderer(canvas, env);
const statusEl = document.getElementById('status');

env.reset();
renderer.draw();

function updateStatus(action, result) {
  if (result.done && result.reward > 0) {
    statusEl.textContent = `Reached the goal! Total reward: ${env.totalReward.toFixed(2)} in ${env.steps} steps. Press R to reset.`;
  } else if (result.done && result.reward < 0) {
    statusEl.textContent = `Fell in a pit! Total reward: ${env.totalReward.toFixed(2)} in ${env.steps} steps. Press R to reset.`;
  } else {
    statusEl.textContent = `Moved ${ACTION_NAMES[action]} | Step ${env.steps} | Reward: ${env.totalReward.toFixed(2)}`;
  }
}

document.addEventListener('keydown', (e) => {
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
      statusEl.textContent = 'Reset. Use arrow keys to move the agent.';
      return;
  }

  if (action === null) return;
  e.preventDefault(); // don't scroll the page

  const result = env.step(action);
  renderer.draw();
  updateStatus(action, result);
});
