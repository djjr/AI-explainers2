# Handoff: gradient descent teaching apps, and a plan for an LLM tutor

_Written 2026-09-19. Status: **the LLM tutor is evaluated, not built.** Nothing in this document has been implemented yet._

This file does two jobs. Part 1 records what exists in this directory so a later session (or Claude Code) can pick up without re-deriving it. Part 2 is the evaluation and plan for wiring an LLM into the apps, in a text version and a voice version.

---

## Part 1. What exists

### Files

| File | What it is |
|---|---|
| `gd_trajectory.py` / `.png` | Matplotlib contour map of gradient descent on the 3-point line-fitting problem, with the 6-step path and a 400-step continuation. |
| `gd_3d.py` / `.html` / `.png` | Plotly 3D loss surface plus the contour view, with camera buttons (angled, top-down). Cells are marked `# %%` so they convert to notebook cells. |
| `gd_app.html` (**v1**) | Self-contained JS app: data space, parameter space with banded contours and the training path, loss curve, readout. Controls on the left. |
| `gd_app_v2.html` (**v2**) | v1 plus a **test set**: a dropdown of "test worlds" (same rule y=2x, shallower y=0.5x, different y=3x−2), a second loss curve, a second minimum marker, optional test-loss rings. |
| `gd_app_v3.html` (**v3**) | **Polynomial features** (degree 1 to 7) with train vs. test: four charts (data space, best fit at each degree, loss during training, weights), a "true rule" dropdown, **Fit to convergence**, and pop-up explainer slides. |

All three apps are single HTML files with no libraries and no network requests. They fill whatever frame they are given (controls left, plots right), follow the OS light/dark setting, and stack vertically below 520px wide.

### The running example

Three points (1,2), (2,4), (5,10). One neuron, identity activation, ŷ = m·x + b, loss = mean squared error. Starting at m = b = 0: loss 40, ∂L/∂m = −40, ∂L/∂b = −32/3 ≈ −10.67. With η = 0.01 the first step gives m = 0.4, b ≈ 0.107, loss ≈ 24.7. After 10 iterations m ≈ 1.702, b ≈ 0.437 (not near the final m = 2, b = 0; the valley is narrow and the last stretch is slow).

### Design decisions worth keeping

- **Chain-rule derivation**, not polynomial expansion: ∂L/∂m = (2/N)Σxᵢeᵢ and ∂L/∂b = (2/N)Σeᵢ. The extra factor of x is "error weighted by influence."
- **Update rule sign is subtract**: w ← w − η·∂L/∂w.
- **v1/v2 "Restart"** returns to where the run began; **"Reset data"** restores the original points *and* puts the model at (0, 0). (An earlier version left the model where it was; that was fixed.)
- **Linear height, not log, on the 3D surface.** A log axis turned the true minimum (loss exactly 0) into a needle.
- **v3 rescales x** to t = (x − 3.25)/2.75 so the features sit near [−1, 1]. Without this, gradient descent on raw powers is hopeless at degree 3 and above (over 200,000 iterations vs. 420).
- **v3 convergence is judged on the weights**, within 0.1% of the closed-form answer, not on the loss, because loss flattens before the weights arrive.
- **v3 stability limit for η** is computed per degree (about 1.0 at degree 1 down to about 0.65 at degree 7), shown under the slider. Default η = 0.30 is stable for every degree.

### Numbers verified (default data, line truth, η = 0.3 unless noted)

- Best-fit test loss by degree: 0.03 (1), 0.03 (2), 0.18 (3), 0.27 (4), 1.28 (5), 1.21 (6), about 62 (7). Degree 7 has 8 parameters for 8 points, so train loss is about 0.
- Iterations to converge: about 21 (deg 1), 420 (deg 3), 14,600 (deg 5), 110,200 (deg 6), 1,875,400 (deg 7). Degree 7 takes about half a second in a modern browser.
- Largest weight at degree 7 after fitting: about 180.
- v2, test worlds (test loss at convergence vs. train): the shallower line y = 0.5x makes test loss rise as train loss falls; y = 3x − 2 leaves a floor (about 3.3 on the original three x values).

### Known limits and open threads

- **Phones:** below 600px the explainer slide becomes a bottom sheet. Testing was done with scripted events in headless Chrome, not on a physical device. On phones the controls come *after* the charts, so students scroll to reach Play and the sliders.
- **v1 loss curve:** ideas not yet built: a log-scale toggle, and faint "ghost" curves from previous runs so learning-rate comparisons are visible.
- **Not built:** a hidden-layer / nonlinear-activation version (the natural next lesson: why depth and nonlinearity exist); regularization (large alternating weights are shown in v3, the remedy is only mentioned).
- **Notebook:** the plan is to move these into the AI 101 Part 1 notebook later. `gd_3d.py` already has `# %%` cell markers. The Part 1 notebook itself is not in this directory.
- **The b = 1.3 discrepancy** from a user run at 10 epochs (correct value is about 0.44) is unresolved; the suspected cause is an inflated b-gradient (a dropped 1/N, or b's update using e·x). Ask to see the loop.

---

## Part 2. Plan: an LLM tutor inside the apps

### Goal

Let a student ask questions about what is on screen ("why did test loss go up?", "what does this weight chart mean?") and get an explanation grounded in the app's actual state, optionally with a live demo. Two versions: **text** first, **voice** later.

### The key idea: send state, not screenshots

The app already knows everything on screen. Each question carries a small structured snapshot. Values below are illustrative.

```json
{
  "app_version": "v3",
  "chart_in_focus": "loss_during_training",
  "degree": 7,
  "true_rule": "line",
  "eta": 0.3,
  "eta_stability_limit": 0.65,
  "iteration": 50000,
  "train_loss": 0.22,
  "test_loss": 6.4,
  "lowest_test_loss": { "iteration": 25, "value": 0.03 },
  "max_abs_weight": 30.4,
  "best_degree_by_test_loss": 1,
  "status": "fitting"
}
```

Why this beats images: it is cheaper, exact, and lets the tutor cite numbers ("your test loss is 6.4 against 0.22 on train"). The system prompt should also include the slide text (the `SLIDES` array in v3) as reference material, so explanations stay consistent with what students have already read.

### Actions (optional, phase 2 of the text version)

Let the tutor propose a small whitelist of actions that map to functions the app already has (the same ones "Show me" uses). The **app validates every action**; the model never touches the page directly.

| Action | Arguments | Existing function |
|---|---|---|
| `set_degree` | integer 1 to 7 | `setDegree` |
| `set_true_rule` | `line` / `curve` / `wave` | truth dropdown, then `fullReset` |
| `set_eta` | number within the slider range | learning-rate slider |
| `fit` | none | Fit to convergence |
| `restart` | none | Restart |
| `spotlight` | chart index 0 to 3 | slide spotlight |

The tutor cannot change data points or anything outside this list.

### Architecture (text version)

```
browser (app)  ──question + state──►  small proxy on a server you control  ──►  Claude API
      ◄──streamed answer (+ optional actions)──
```

- **The API key lives only in the proxy.** It can never be in the HTML.
- **The proxy** applies a fixed system prompt, caps input and output length, rate-limits per session, enforces a daily budget, and streams the response.
- **Embedding:** the apps currently make no network requests. Once they call the proxy, the host page must allow that request (CORS on the proxy; CSP or sandbox settings on wherever the iframe is embedded, including any LMS). Test this early.
- **UI:** an "Ask" mode inside the existing explainer sheet, so it inherits dragging, collapsing, and the spotlight ring. Show the state snapshot in a collapsed "what I can see" line so students know what the tutor is looking at.

### System prompt principles (the hard part is pedagogy, not plumbing)

- **Predict first.** Default to asking the student what they expect before explaining, then confirm or correct using the live numbers.
- **Short, concrete, grounded.** Cite the actual values from the snapshot. Do not invent numbers.
- **Stay in scope.** This app's concepts (model, loss, gradient, learning rate, overfitting, train vs. test). Decline or redirect otherwise.
- **Admit uncertainty.** If the snapshot doesn't contain what is needed, say so.
- **Tone and length:** brief by default; offer to go deeper.

### Evaluation before students see it

- Write 30 to 50 realistic student questions (including confused and wrong ones). Grade answers for correctness against the verified numbers above.
- Red-team it: attempts to get it off-topic, to get answers to assignments, to make it emit unlisted actions.
- Include the **error-catching assignment** idea: students find and document a case where the tutor is wrong. This turns the tool into course material on expert asymmetry (a system whose reasoning you can't fully audit), which is a course theme.

### Voice version

Build on top of the text version, not instead of it.

| Piece | Approach and caveats |
|---|---|
| Speech in | Start with browser speech recognition (free, but uneven across browsers, and it mangles terms like "gradient" and "η"). Hosted recognition is more accurate and costs money. |
| Speech out | Browser text-to-speech first (robotic but free). Hosted voices sound natural but add a vendor. |
| Embedding | An iframe must be granted microphone permission explicitly (`allow="microphone"`). Check this in slides and LMS embeds. |
| Latency | Expect roughly 2 to 4 seconds end to end. Stream the answer and start speaking at sentence boundaries. |
| Turn-taking | **Use push-to-talk.** It removes most of the hard problems: interruption, speaker-to-mic echo, deciding when someone has finished. |
| Content | Spoken answers must be shorter and free of formulas and lists. Keep live captions for accessibility and noisy rooms. |

**Not verified:** whether Anthropic offers a native speech API. Assume the pipeline above (recognition, then the model, then speech) and check the current documentation before committing. Likewise, current API pricing, rate limits, and data-retention terms have not been checked and should be confirmed.

### Audience decides the priority

- **Self-study (students on their own):** text first, voice as a bonus.
- **Live in a lecture room:** voice is a poor fit for many simultaneous users. It could work as a single on-stage demo, but it is risky live, so keep a text fallback.

### Privacy and policy

Student questions, and especially audio, leave the device. Before any student use: check university data-handling rules; default to logging nothing (or only anonymous aggregates); collect no names; state clearly in the UI what is sent.

### Suggested phases

1. **Prototype (text, no actions).** Proxy plus state packet plus streamed answers in the sheet. Test with 5 to 10 people.
2. **Evaluate.** The question set and red-team above. Fix the system prompt.
3. **Add whitelisted actions**, with validation and a visible "the tutor changed X" notice.
4. **Voice prototype** with push-to-talk and browser speech. Measure whether students use it.
5. **Upgrade the speech layer** only if usage justifies it.

### Open questions for the owner

1. Primary audience: self-study, or live in the room?
2. Where will the proxy be hosted, and who pays for API usage (a fixed budget and a per-session cap)?
3. Logging policy: nothing, anonymous aggregates, or full transcripts (needs consent)?
4. Which apps get it first: v3 only, or all three?
5. Is the error-catching assignment in scope for the course?
6. Should the Ask mode require a course login, to prevent outside use?

---

## How to resume

- Start a new session with this file plus the three `gd_app*.html` files.
- To begin the text prototype, ask for the proxy and the state-packet function first (a `getSnapshot()` in v3 that returns the JSON above). The app's `st` object already holds every field.
- Keep the apps single-file and dependency-free; the proxy call is the only new network dependency.
