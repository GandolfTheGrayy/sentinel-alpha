import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const FINNHUB = 'https://finnhub.io/api/v1';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body || {};
  const { portfolio_id, action, thesis, prediction_id } = body;
  const ticker = String(body.ticker || '').toUpperCase().trim();
  const shares = Number(body.shares);

  if (!['agent', 'human'].includes(portfolio_id)) return res.status(400).json({ error: 'portfolio_id must be "agent" or "human"' });
  if (!['buy', 'sell'].includes(action)) return res.status(400).json({ error: 'action must be "buy" or "sell"' });
  if (!/^[A-Z][A-Z.\-]{0,7}$/.test(ticker)) return res.status(400).json({ error: 'invalid ticker format' });
  if (!Number.isFinite(shares) || shares <= 0) return res.status(400).json({ error: 'shares must be a positive number' });

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  const finnhubKey = process.env.FINNHUB_API_KEY;
  if (!url || !key || !finnhubKey) return res.status(500).json({ error: 'missing keys' });

  const supa = createClient(url, key, { auth: { persistSession: false } });
  try {
    const qRes = await fetch(`${FINNHUB}/quote?symbol=${ticker}&token=${finnhubKey}`);
    const quote: any = await qRes.json();
    if (!quote?.c) return res.status(404).json({ error: 'ticker not found on Finnhub' });
    const price = Number(quote.c);

    const { data: portfolio, error: pErr } = await supa.from('portfolios').select('*').eq('id', portfolio_id).single();
    if (pErr || !portfolio) return res.status(500).json({ error: 'portfolio not found' });

    if (action === 'buy') {
      const cost = shares * price;
      if (cost > Number(portfolio.cash)) {
        return res.status(400).json({ error: `insufficient cash: need $${cost.toFixed(2)}, have $${Number(portfolio.cash).toFixed(2)}` });
      }
      await supa.from('portfolios').update({ cash: Number(portfolio.cash) - cost }).eq('id', portfolio_id);
      await supa.from('positions').insert({
        portfolio_id, ticker, shares, entry_price: price,
        thesis: thesis || null, prediction_id: prediction_id || null,
      });
      await supa.from('trades').insert({
        portfolio_id, ticker, action: 'buy', shares, price,
        thesis: thesis || null, prediction_id: prediction_id || null,
      });
      return res.status(200).json({ ok: true, action, ticker, shares, price, cost });
    }

    // SELL — FIFO
    const { data: openLots } = await supa
      .from('positions')
      .select('*')
      .eq('portfolio_id', portfolio_id)
      .eq('ticker', ticker)
      .eq('closed', false)
      .order('entry_time', { ascending: true });
    if (!openLots || !openLots.length) return res.status(400).json({ error: `no open ${ticker} position` });

    const totalOpen = openLots.reduce((s: number, l: any) => s + Number(l.shares), 0);
    if (shares > totalOpen + 1e-9) {
      return res.status(400).json({ error: `only ${totalOpen} shares of ${ticker} held, requested ${shares}` });
    }

    let remaining = shares;
    let proceeds = 0;
    for (const lot of openLots) {
      if (remaining <= 1e-9) break;
      const lotShares = Number(lot.shares);
      if (lotShares <= remaining + 1e-9) {
        await supa.from('positions').update({
          closed: true,
          exit_price: price,
          exit_time: new Date().toISOString(),
          exit_reason: 'manual sell',
        }).eq('id', lot.id);
        proceeds += lotShares * price;
        remaining -= lotShares;
      } else {
        await supa.from('positions').update({ shares: lotShares - remaining }).eq('id', lot.id);
        proceeds += remaining * price;
        remaining = 0;
      }
    }

    await supa.from('portfolios').update({ cash: Number(portfolio.cash) + proceeds }).eq('id', portfolio_id);
    await supa.from('trades').insert({
      portfolio_id, ticker, action: 'sell', shares, price,
      thesis: thesis || null, prediction_id: prediction_id || null,
    });
    return res.status(200).json({ ok: true, action, ticker, shares, price, proceeds });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'trade failed' });
  }
}
