import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const strategy = String(req.query.strategy || '').trim();
  const status = String(req.query.status || 'all').trim();
  const limit = Math.min(parseInt(String(req.query.limit || '100'), 10) || 100, 500);
  const ticker = String(req.query.ticker || '').toUpperCase().trim();

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_KEY;
  if (!url || !key) return res.status(500).json({ error: 'SUPABASE keys missing' });

  const supa = createClient(url, key, { auth: { persistSession: false } });
  try {
    let q = supa.from('predictions').select('*').order('made_on', { ascending: false }).limit(limit);
    if (strategy) q = q.eq('strategy', strategy);
    if (status === 'open') q = q.eq('resolved', false);
    if (status === 'resolved') q = q.eq('resolved', true);
    if (ticker) q = q.eq('ticker', ticker);
    const { data, error } = await q;
    if (error) throw error;
    res.setHeader('Cache-Control', 'no-store');
    return res.status(200).json({ predictions: data || [] });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'predictions fetch failed' });
  }
}
