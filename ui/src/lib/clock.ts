/**
 * Single time source for the UI.
 *
 * By default `nowMs()` is the wall clock. When the backend runs in sim mode it reports
 * `engine.sim_time` on every status tick; AppContext forwards it here and `nowMs()` then
 * returns that virtual instant advanced by the real milliseconds elapsed since the tick
 * arrived, so clocks and countdowns keep moving between ticks. Every relative time,
 * countdown, age column and the top-bar ET clock read this source.
 */

let simBaseMs: number | null = null
let simReceivedAt = 0

/** Record the latest simulated time (ISO UTC) or clear it with null/undefined. */
export function setSimTime(iso: string | null | undefined): void {
  if (!iso) {
    simBaseMs = null
    return
  }
  const ms = Date.parse(iso)
  if (!Number.isFinite(ms)) {
    simBaseMs = null
    return
  }
  simBaseMs = ms
  simReceivedAt = Date.now()
}

/** True while the backend reports a simulated clock. */
export function isSimClock(): boolean {
  return simBaseMs !== null
}

/** Current time in ms: simulated (locally advanced) when available, else the wall clock. */
export function nowMs(): number {
  if (simBaseMs === null) return Date.now()
  return simBaseMs + (Date.now() - simReceivedAt)
}
