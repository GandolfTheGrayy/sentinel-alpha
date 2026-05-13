import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESERVED = new Set(['agent', 'human']);
const STARTING_DEFAULT = 1000.0;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body || {};
  const id = String(body.id || '').toLowerCase().trim();
  const name = String(body.name || '').trim();
  const description = String(body.description || '').trim();
  const starting = Number(body.starting_cash || STARTING_DEFAULT);

  if (!/^[a-z0-9_]{2,24}$/.test(id)) return res.status(400).json({ error: 'id must be 2-24 chars, lowercase letters/digits/underscores' });
  if (RESERVED.has(id)) return res.status(400).json({ error: `id "${id}" is reserved` });
  if (!Number.isFinite(starting) || starting < 50 || starting > 1_000_000) return res.status(400).json({ error: 'starting_cash must be 50–1,000,000' });

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  if (!url || !key) return res.status(500).json({ error: 'SUPABASE keys missing' });
  const supa = createClient(url, key, { auth: { persistSession: false } });

  try {
    const { data: existing } = await supa.from('portfolios').select('id').eq('id', id).maybeSingle();
    if (existing) return res.status(409).json({ error: 'portfolio id already exists' });

    const { error } = await supa.from('portfolios').insert({
      id,
      cash: starting,
      starting_cash: starting,
      name: name || id,
      description: description || null,
    });
    if (error) throw error;
    return res.status(200).json({ ok: true, id, starting_cash: starting });
  } catch (e: any) {
    return res.status(500).json({ error: e?.message || 'create failed' });
  }
}
