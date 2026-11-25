## Lab memory (as of 2026-11-18, analyst session)
### Integrity / tooling (unresolved, needs a human)
- Equity is -14.0% since start (-11.5% on 11/06). R and P&L still disagree: meanrev_bb#2 +1.06R but P&L -224 and PF 0.84; orb#2 +0.59R but PF 1.6 and P&L +209. Prefer evidence where PF, P&L and R agree.
- Stop exits below -1R (-2 to -6R on trend_ema holds >1000 min) suggest gaps or an R-accounting bug. Needs an audit.
- The backtester has about 30 days of bars, so walk-forward segment 1 is empty. `--markets` is ignored with `--variant`. Run one backtest per command; loops and pipes are denied.
- The meanrev_bb#2 NVDA +196R outlier contaminates all-time and attribution stats. Ignore the high buckets (rsi_2 low, bb_pos low, vol_ratio high, Tue 09h, range/high_vol, signal exit) and the OOS filter candidates built on them.
- The allocator needs a minimum-sample gate. trend_ema#1 and #2 hold about 0.17 each with CIs spanning zero and negative P&L.
- Filters on regime features can fail the 15-trade backtest gate. Only propose filters that leave enough trades.
### Controls
- Live random entry: -0.53R (n=117).
### Evidence
- breakout_orb#1 (n=28, +0.61R, CI +0.04..+1.14, PF 7.0, P&L +1025) and orb#2 (n=26, +0.59R, CI +0.10..+1.13) are the only variants beating the control with a non-overlapping CI. Watch at n>=30.
- meanrev_bb#3 is marginal: +0.10R, PF 1.35, n=64, CI spans zero.
- trend_ema is about -0.09R at family level. trend_ema#4 was accepted on a backtest (OOS +0.20R) but is -0.28R live (n=27), so backtest acceptance has weak predictive value here.
- squeeze is the worst family (-0.40R). A retire proposal for squeeze#2 was made on 11/18. squeeze#1 is on probation.
- vwap_revert#1 has no edge (-0.05R, n=163).
- Crypto is -0.45R and equities +0.23R. Stops are 503 trades at about -1R, the structural cost.
### Hypotheses
- H1: Crypto exclusion helps fee-bound variants. Supported, but a filter has not been tested on trend_ema.
- H2: For meanrev, a slower timeframe and more extreme entries are better. Weak.
- H3: A ret_20 cap on mean reversion is untested.
- H5: Excluding low-vol regimes helps. Untested because of backtest-gate failures.
- H6: orb variants with r_mult or or_minutes tweaks may improve. Untested.
### Next session
- Run backtests, with the budget available: crypto exclusion for trend_ema#1/#2; orb#1 param variations; meanrev_bb#3 params.
- Check squeeze#2 retirement outcome. Consider retiring squeeze#1 if it stays negative.
- No new families until accounting is fixed.
