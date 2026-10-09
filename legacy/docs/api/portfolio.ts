import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const FINNHUB = 'https://finnhub.io/api/v1';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const id = String(req.query.id || '').trim();
  if (!['agent', 'human'].includes(id)) {
    return res.status(400).json({ error: 'id must be "agent" or "human"' });
  }
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  const finnhubKey = process.env.FINNHUB_API_KEY;
  if (!url || !key || !finnhubKey) {
    return res.status(500).json({ error: 'missing keys (SUPABASE / FINNHUB)' });
  }

  const supa = createClient(url, key, { auth: { persistSession: false } });
  try {
    const [pRes, posRes, trRes] = await Promise.all([
      supa.from('portfolios').select('*').eq('id', id).single(),
      supa.from('positions').select('*').eq('portfolio_id', id).order('entry_time', { ascending: false }),
      supa.from('trades').select('*').eq('portfolio_id', id).order('created_at', { ascending: false }).limit(30),
    ]);
    if (pRes.error) throw pRes.error;
    const portfolio: any = pRes.data;
    const positions: any[] = posRes.data || [];
    const trades: any[] = trRes.data || [];

    const open = positions.filter(p => !p.closed);
    const closed = positions.filter(p => p.closed).slice(0, 10);

    const tickers = [...new Set(open.map(p => p.ticker))];
    const priceMap: Record<string, number> = {};
    await Promise.all(tickers.map(async (t) => {
      try {
        const r = await fetch(`${FINNHUB}/quote?symbol=${t}&token=${finnhubKey}`);
        const q: any = await r.json();
        if (q?.c) priceMap[t] = q.c;
      } catch {}
    }));

    const enrichedOpen = open.map((p: any) => {
      const cur = priceMap[p.ticker];
      const cost = Number(p.shares) * Number(p.entry_price);
      const value = cur != null ? Number(p.shares) * cur : null;
      const pnl = value != null ? value - cost : null;
      const pnl_pct = pnl != null && cost > 0 ? (pnl / cost) * 100 : null;
      return { ...p, current_price: cur ?? null, value, pnl, pnl_pct };
    });
    const enrichedClosed = closed.map((p: any) => {
      const cost = Number(p.shares) * Number(p.entry_price);
      const proceeds = Number(p.shares) * Number(p.exit_price ?? 0);
      const pnl = proceeds - cost;
      const pnl_pct = cost > 0 ? (pnl / cost) * 100 : null;
      return { ...p, pnl, pnl_pct };
    });

    const positions_value = enrichedOpen.reduce((s: number, p: any) => s + (p.value || 0), 0);
    const total_value = Number(portfolio.cash) + positions_value;
    const total_return_pct = ((total_value - Number(portfolio.starting_cash)) / Number(portfolio.starting_cash)) * 100;

    res.setHeader('Cache-Control', 'no-store');
    return res.status(200).json({
      ...portfolio,
      open_positions: enrichedOpen,
      closed_positions: enrichedClosed,
      trades,
      positions_value,
      total_value,
      total_return_pct,
    });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'portfolio fetch failed' });
  }
}
