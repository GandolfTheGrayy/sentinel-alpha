import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const FINNHUB = 'https://finnhub.io/api/v1';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body || {};
  const position_id = Number(body.position_id);
  if (!Number.isFinite(position_id) || position_id <= 0) return res.status(400).json({ error: 'position_id required' });

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  const finnhubKey = process.env.FINNHUB_API_KEY;
  if (!url || !key || !finnhubKey) return res.status(500).json({ error: 'missing keys' });

  const supa = createClient(url, key, { auth: { persistSession: false } });
  try {
    const { data: pos, error: pErr } = await supa.from('positions').select('*').eq('id', position_id).single();
    if (pErr || !pos) return res.status(404).json({ error: 'position not found' });
    if (pos.closed) return res.status(400).json({ error: 'position already closed' });

    const qRes = await fetch(`${FINNHUB}/quote?symbol=${pos.ticker}&token=${finnhubKey}`);
    const quote: any = await qRes.json();
    if (!quote?.c) return res.status(404).json({ error: 'ticker quote unavailable' });
    const price = Number(quote.c);
    const shares = Number(pos.shares);
    const proceeds = shares * price;

    await supa.from('positions').update({
      closed: true,
      exit_price: price,
      exit_time: new Date().toISOString(),
      exit_reason: 'manual close',
    }).eq('id', position_id);

    const { data: portfolio } = await supa.from('portfolios').select('cash').eq('id', pos.portfolio_id).single();
    if (portfolio) {
      await supa.from('portfolios').update({ cash: Number(portfolio.cash) + proceeds }).eq('id', pos.portfolio_id);
    }
    await supa.from('trades').insert({
      portfolio_id: pos.portfolio_id,
      ticker: pos.ticker,
      action: 'sell',
      shares,
      price,
      thesis: 'manual close',
      prediction_id: pos.prediction_id,
    });

    return res.status(200).json({ ok: true, ticker: pos.ticker, shares, price, proceeds });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'close failed' });
  }
}
