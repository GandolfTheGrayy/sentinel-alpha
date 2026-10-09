# Sentinel lab digest - 2026-10-02 20:45 UTC (analyst session)
Mode **sim** | equity $99,758 | today -242 (-0.24%) | week -226 | since start -240 (-0.24%) | open lots 23
Population: 16 active, 0 incubating, 0 probation, 0 retired (cap 24). Claude budget this week: $0.00 of $62.50.

## Since the last analyst session (2026-10-01T20:45 UTC)
- 48 trades closed

## Tournament (all-time on closed trades; last30 = last 30 trades)
| variant | status | alloc | n | expR (95% CI) | win | PF | last30 expR | maxDD% | P&L | origin |
|---|---|---|---|---|---|---|---|---|---|---|
| meanrev_bb#1 | active | 0.20 | 9 | +0.33 (-0.22..+0.95) | 67% | 3.2 | +0.33 (n=9) | 0.4 | +173 | seed |
| vwap_revert#1 | active | 0.03 | 10 | +0.03 (-0.79..+0.76) | 40% | 0.851 | +0.03 (n=10) | 2.5 | -23 | seed |
| meanrev_bb#2 | active | 0.03 | 18 | -0.17 (-0.56..+0.25) | 44% | 0.667 | -0.17 (n=18) | 4.6 | -92 | seed |
| trend_ema#1 | active | 0.20 | 2 | -0.54 (-0.54..-0.54) | 0% | 0.0 | -0.54 (n=2) | 0.4 | -77 | seed |
| random_entry#1 | active | 0.01 | 2 | -0.57 (-0.57..-0.57) | 0% | 0.0 | -0.57 (n=2) | 0.6 | -13 | seed |
| breakout_orb#1 | active | 0.03 | 2 | -1.06 (-1.06..-1.06) | 0% | 0.0 | -1.06 (n=2) | 2.4 | -78 | seed |
| squeeze#2 | active | 0.03 | 1 | -1.06 (-1.06..-1.06) | 0% | 0.0 | -1.06 (n=1) | 2.2 | -71 | seed |
| squeeze#1 | active | 0.03 | 4 | -1.08 (-1.08..-1.08) | 0% | 0.0 | -1.08 (n=4) | 6.9 | -226 | seed |
| trend_ema#2 | active | 0.03 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| breakout_orb#2 | active | 0.03 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| breakout_donchian#1 | active | 0.03 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| breakout_donchian#2 | active | 0.03 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| xs_momentum#1 | active | 0.03 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| overnight#1 | active | 0.03 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| crypto_mtf#1 | active | 0.20 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |
| buy_hold#1 | active | 0.01 | 0 | n/a (+0.00..+0.00) | n/a | n/a | n/a (n=0) | 0.0 | +0 | seed |

Control (random entries, same exits): n=2, expectancy -0.57R, win 0%. A family is only interesting if it beats this with a non-overlapping confidence interval.

## Factor attribution
**Attribution window:** last 45 days, 48 trades, expectancy -0.20R (95% CI -0.49..+0.09), win rate 38%, P&L -407.

**By family:** meanrev_bb n=27 -0.00R wr=52%; vwap_revert n=10 +0.03R wr=40%; squeeze n=5 -1.08R wr=0%; breakout_orb n=2 -1.06R wr=0%; random_entry n=2 -0.57R wr=0%; trend_ema n=2 -0.54R wr=0%

**By market:** equities n=43 -0.14R wr=42%; crypto n=5 -0.67R wr=0%

**By regime:** range/mid_vol n=14 -0.72R wr=7%; range/high_vol n=11 +0.15R wr=46%; trend/low_vol n=8 -0.24R wr=38%; range/low_vol n=6 -0.20R wr=50%; trend/mid_vol n=5 +0.62R wr=80%; trend/high_vol n=4 -0.28R wr=50%

**By exit reason:** stop n=25 -1.05R wr=0%; target n=7 +1.44R wr=100%; signal n=6 +0.56R wr=100%; time n=6 +0.40R wr=50%; session_end n=4 +0.20R wr=50%

**By side:** long n=37 -0.34R wr=30%; short n=11 +0.29R wr=64%

**Top features** (abs. Spearman correlation with R-multiple):
- `rvol_20d` imp=0.220 -> 0.147..0.29: -0.08R (n=16), 0.29..0.383: -0.04R (n=17), 0.383..0.708: -0.51R (n=15)
- `sleeve_dd_pct` imp=0.201 -> -0.001..0.533: -0.34R (n=32), 0.533..1.96: +0.09R (n=16)
- `regime_vol` imp=0.195 -> low vol: -0.23R (n=14), mid vol: -0.37R (n=19), high vol: +0.04R (n=15)
- `mkt_rvol_20d` imp=0.190 -> 0.335007: -0.14R (n=43), 0.576296: -0.67R (n=5)
- `is_crypto` imp=0.190 -> equities: -0.14R (n=43), crypto: -0.67R (n=5)
- `session` imp=0.189 -> closed: -0.58R (n=2), pre: -0.73R (n=3), regular: -0.14R (n=43)
- `mkt_ret_5d` imp=0.166 -> -0.0646..-0.0249: -0.44R (n=18), -0.0249..-0.021: -0.41R (n=14), -0.021..-0.0117: +0.26R (n=16)
- `mkt_dist_sma50_pct` imp=0.166 -> -12.7..-3.92: -0.44R (n=18), -3.92..-3.54: -0.41R (n=14), -3.54..-2.62: +0.26R (n=16)
- `mkt_ret_1d` imp=0.158 -> -0.0276..-0.0058: -0.44R (n=18), -0.0058..-0.00184: -0.41R (n=14), -0.00184..0.00764: +0.26R (n=16)
- `rsi_2` imp=0.152 -> 0.0328..3.78: -0.38R (n=16), 3.78..71.5: -0.05R (n=16), 71.5..99.8: -0.17R (n=16)

**Filter candidates (validated out of sample):**
- exclude rvol_20d in 0.383..0.708: removes 15 trades, expectancy -0.20R -> -0.06R, OOS delta +0.07R

**Time-of-week:** best Fri 13h n=9 +0.01R | worst Fri 14h n=9 -0.72R

## Notable trades (last 7 days)
- WIN vwap_revert#1 long MSFT +1.97R (target, 44 min) - -2.9 ATR below VWAP on vol x1.3
- WIN vwap_revert#1 long GLD +1.62R (time, 120 min) - -3.6 ATR below VWAP on vol x1.3
- WIN meanrev_bb#1 short NFLX +1.58R (target, 97 min) - close above BB upper, RSI2 98
- LOSS meanrev_bb#2 long NFLX -1.27R (stop, 21 min) - close below BB lower, RSI2 2
- LOSS vwap_revert#1 long AMZN -1.22R (stop, 74 min) - -3.3 ATR below VWAP on vol x1.5
- LOSS meanrev_bb#2 long AMZN -1.17R (stop, 14 min) - close below BB lower, RSI2 1

## Market regime
- SPY: 5d +0.4%, 20d -1.7%, realised vol 34% ann., ADX14 14, vs SMA50 -3.4%
- BTC/USD: 5d -7.9%, 20d -7.7%, realised vol 48% ann., ADX14 21, vs SMA50 -10.3%

## Families available
- `breakout_donchian` (equities/crypto, 1h): Donchian channel breakout with a chandelier-style ATR trailing stop; works on equities and crypto. Params: tf:1h/4h/15m, n:10-60, stop_atr:1.0-4.0, trail_atr:1.5-6.0, max_hold:24-400, min_atr_pct:0.0-1.5
- `breakout_orb` (equities, 5m): Opening-range breakout: trade the first break of the opening N-minute range with volume confirmation; flat by the close. Params: or_minutes:15/30/60, latest_entry_min:60-300, vol_mult:0.8-2.5, r_mult:1.0-4.0, stop_atr:0.8-3.0, max_hold:12-78
- `buy_hold` (equities/crypto, 1d, CONTROL): CONTROL: buy-and-hold the benchmark of each market with a very wide stop. The passive baseline. Params: none
- `crypto_mtf` (crypto, 15m): Long-only crypto: 4h EMA trend up, wait for a 15m RSI pullback, enter on the first up-close; ATR stop, R target and trailing. Params: trend_ema:20-120, fast_ema:8-40, rsi_pullback:20.0-50.0, stop_atr:1.0-4.0, r_mult:1.5-6.0, trail_atr:1.5-6.0, max_hold:24-600
- `meanrev_bb` (equities, 15m): Fade closes outside the Bollinger bands when RSI(2) is extreme; target the middle band; optional daily trend filter. Params: tf:5m/15m, bb_n:10-40, bb_k:1.5-3.0, rsi_lo:3.0-25.0, rsi_hi:75.0-97.0, stop_atr:0.8-3.0, max_hold:4-60, trend_filter:0/1
- `overnight` (equities, 1d): Buy strong names shortly before the close and sell shortly after the next open to harvest the overnight premium. Params: n_symbols:1-8, trend_sma:20-200, max_day_move_pct:1.0-6.0, stop_atr:1.0-5.0, exit_after_open_min:1-60, rank_lookback:5-60
- `random_entry` (equities/crypto, 15m, CONTROL): CONTROL: random entries with the standard ATR stop / R target / time exits. Any real family must beat this. Params: none
- `squeeze` (equities/crypto, 15m): After at least k bars of Bollinger bands inside the Keltner channel, enter when the bands release, in the direction of momentum. Params: tf:15m/1h, min_squeeze:3-20, kc_mult:1.0-2.5, bb_k:1.5-2.5, stop_atr:0.8-3.0, r_mult:1.0-4.0, max_hold:10-200
- `trend_ema` (equities/crypto, 15m): Enter on an EMA fast/slow crossover when ADX confirms a trend; ride it with an ATR trailing stop. Params: tf:15m/1h, fast:5-30, slow:20-120, adx_min:12.0-40.0, stop_atr:1.0-4.0, trail_atr:1.5-5.0, max_hold:20-300
- `vwap_revert` (equities, 5m): When price is stretched more than z ATRs from the session VWAP on a volume climax, fade it with VWAP as the target. Params: tf:5m/15m, z:1.0-4.0, vol_mult:0.8-3.0, min_after_open:5-120, stop_atr:0.8-3.0, max_hold:6-60, latest_entry_min:120-360
- `xs_momentum` (equities, 1d): Each morning rank the equity universe by medium-term return (skipping the last few days) and hold the top names; rotate out when they drop from the top set. Params: lookback:20-160, skip:0-10, n_long:2-8, stop_atr:1.5-6.0, abs_filter:0/1

## Lab memory (your notes from earlier sessions)
_(empty - this is the first session)_