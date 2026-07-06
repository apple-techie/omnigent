# Auto-respawn an orphaned host-bound runner instead of waiting for the next message

## Problem (gap behind #1857 / #1953)

When a **host-bound** runner's tunnel drops unexpectedly — the runner
process died, but the host's `omnigent host` tunnel is still online —
the session is left orphaned. `_on_runner_disconnect` marks it `failed`
and the Subagents panel shows the grey "Agent disconnected" dot, but
nothing relaunches the runner. Recovery only happens on the **next user
message**, which lands in the message-dispatch relaunch path
(`_launch_runner_on_host` → wait for tunnel). Until the user sends
something, the open session just sits there disconnected — the "click to
reconnect" papercut.

## What this does

Adds a **conservative, host-bound-only auto-respawn**: when a runner
disconnects, `_on_runner_disconnect` schedules a bounded background task
that — *only if the central server still holds a live host tunnel for
that conversation* — mints a fresh runner on the host and lets the
existing reconnect machinery take over. The session heals on its own.

It deliberately reuses the paths that already exist rather than
duplicating them:

- `_launch_runner_on_host()` mints + binds the replacement runner (it
  already calls `replace_runner_id`);
- `_wait_for_runner_client()` waits for the new tunnel;
- the existing `_on_runner_connect` callback then re-POSTs `/v1/sessions`,
  restarts the SSE relay, and clears the stale `runner_disconnected`
  status via `_publish_runner_recovered_status`. **None of that recovery
  logic is duplicated here.**

## Safety mitigations (the contract, not extras)

| Mitigation | Implementation |
|---|---|
| **(a) Host-bound only** | Gated on the in-memory `host_registry.get(conv.host_id)` returning a live tunnel — **not** the DB host-liveness row. Only a replica actually holding the host tunnel can send `host.launch_runner`. A session with no `host_id` (CLI / `local_stranded`) or whose host tunnel isn't live on this replica is left to today's manual path, untouched. |
| **(b) Debounce** | The task first waits `_AUTO_RESPAWN_DEBOUNCE_S` (5s) via `tunnel_registry.wait_for_runner(dead_runner_id)`. A transient WS flap that the runner client reconnects on its own returns the same runner → the respawn **aborts**. Only a genuinely dead runner is replaced. |
| **(c) Respect intentional Stop** | Stop is non-sticky (it drops the runner tunnel, firing `_on_runner_disconnect` indistinguishably from a crash). The `stop_session` branch now calls `mark_runner_intentionally_stopped(runner_id)` **before** dropping the tunnel; the respawn path consults a short-TTL marker and skips those runners. Auto-respawn is for *unexpected* death only. |
| **(d) Idempotency / no-dup** | A per-conversation `_auto_respawn_in_flight` guard ensures at most one replacement runner is launched per conversation at a time. A second disconnect for an already-rebound runner also finds the row no longer pinned to the dead id and no-ops. |
| **(e) Bounded** | A rolling-window rate limit (`_AUTO_RESPAWN_MAX_ATTEMPTS = 3` per `_AUTO_RESPAWN_WINDOW_S = 300s` per conversation). A host that keeps killing its runner exhausts the budget and falls back to the manual path instead of looping forever. |

The background tasks are strong-referenced (`_auto_respawn_tasks`) and
cancelled at ASGI shutdown via a new `cancel_auto_respawn_tasks()` hook,
mirroring `cancel_managed_launch_tasks()`.

## Out of scope (unchanged)

- **`local_stranded` auto-heal** — needs a resident user-machine daemon
  we don't have; those sessions have no live host tunnel and are left to
  the manual path.
- **The host daemon itself dying (#1857 / #1953, host-side)** — see the
  gap note below.
- **Any UI redesign.** Backend only.

## Files changed

- `omnigent/server/routes/sessions.py` — the auto-respawn module
  (`schedule_runner_auto_respawn`, `_auto_respawn_runner_after_disconnect`,
  `_maybe_respawn_conversation_runner`, the intentional-stop marker,
  the retry budget, `cancel_auto_respawn_tasks`), plus the
  `mark_runner_intentionally_stopped` call in the `stop_session` branch.
- `omnigent/server/app.py` — wires `schedule_runner_auto_respawn` into
  `_on_runner_disconnect` and `cancel_auto_respawn_tasks` into lifespan
  teardown.
- `tests/server/routes/test_runner_auto_respawn.py` — new focused tests.

## Tests

`tests/server/routes/test_runner_auto_respawn.py` covers, at minimum:

- host online → auto-respawn happens (runner rebound, budget consumed);
- host offline (no live tunnel on this replica) → no respawn, manual
  path preserved, no budget consumed;
- not host-bound (`host_id is None`) → no respawn;
- user Stopped → no respawn;
- binding already moved on → no respawn;
- transient flap (original runner reconnects during debounce) → no
  duplicate runner;
- concurrent-guard (respawn already in flight) → no duplicate;
- retry budget exhausted → capped;
- harness-not-configured refusal → no retry loop, guard released;
- multiple bound conversations → each respawned;
- scheduler inline gates (no host registry / intentionally stopped →
  no task scheduled);
- intentional-stop marker TTL expiry.

## Gate commands run and results

| Gate | Command | Result |
|---|---|---|
| Server suite | `env -u PYTHONPATH .venv/bin/python -m pytest tests/server` | **2093 passed, 1 skipped, 3 xfailed** |
| Runner suite | `env -u PYTHONPATH .venv/bin/python -m pytest tests/runner` | 1143 passed, 1 skipped, 4 xfailed, **8 pre-existing environmental failures** (see below) |
| Onboarding suite | `env -u PYTHONPATH .venv/bin/python -m pytest tests/onboarding` | **705 passed** |
| New tests | `... pytest tests/server/routes/test_runner_auto_respawn.py` | **15 passed** |
| Lint | `... ruff check` (+ `ruff format --check`) | clean |
| Pre-commit | `pre-commit run` on staged files | all hooks Passed / Skipped |

The 8 runner failures are all in `tests/runner/test_app_sessions_native.py`
(codex-native: "Terminal registry not configured" / "Codex-native model
options are not ready yet"). They reproduce **identically on the clean
tree with this branch's changes stashed**, and this change touches no
runner code — they are pre-existing environmental failures, not a
regression from this PR.

## Where the #1857 / #1953 host-side death still leaves a gap

This change can only recover a runner death **when the host itself is
still alive and its tunnel is live on the replica handling the
disconnect.** It cannot cover:

1. **The host daemon dying (#1857 / #1953).** If `omnigent host` exits,
   `host_registry.get(conv.host_id)` returns `None`, so we intentionally
   do nothing — there is no tunnel to send `host.launch_runner` over. A
   managed sandbox can be relaunched (the existing
   `_maybe_relaunch_managed_sandbox` path, on the next message), but an
   external/laptop host that goes down needs the user to bring the daemon
   back. That is inherently host-side and out of scope here.
2. **`local_stranded` / CLI sessions.** No `host_id`, no host to relaunch
   on. Needs a resident user-machine daemon we don't have.
3. **A different replica holding the tunnel.** Auto-respawn only fires on
   the replica whose in-memory registry holds the host tunnel. The
   disconnect callback runs on the replica the runner tunnel was pinned
   to, which is normally the same replica — but in a split scenario the
   session falls back to the manual next-message path (which re-resolves
   routing), so it is degraded, not broken.

## Design tension with open PR #1462

PR **#1462** reframes disconnect as a **liveness** signal rather than a
respawn trigger — i.e. it leans toward *reporting* a runner as
down/reconnectable and letting the client/user drive recovery, instead of
the server proactively minting a replacement. This PR takes the opposite
stance for the **host-bound-with-live-host** case specifically: the
server has everything it needs to heal the session, so it does, subject
to the safety gates above.

The two are not mutually exclusive — #1462's liveness model still governs
the cases this PR deliberately does **not** touch (`local_stranded`,
host-offline, cross-replica), where the honest answer is "report it, the
server can't fix it." A maintainer should weigh whether host-bound
auto-respawn belongs in the server (this PR) or whether disconnect should
stay purely a liveness signal (#1462) with recovery always client-driven.
If #1462 lands first, this PR's respawn should slot in *behind* its
liveness reporting (respawn as the host-bound optimization on top of an
honest liveness signal), not replace it.
