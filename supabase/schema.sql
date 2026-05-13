-- Sentinel Alpha — Supabase schema
-- Run this once in the Supabase SQL Editor.

create table if not exists portfolios (
  id text primary key,
  cash numeric not null default 1000.00,
  starting_cash numeric not null default 1000.00,
  created_at timestamptz default now()
);

create table if not exists positions (
  id bigserial primary key,
  portfolio_id text not null references portfolios(id) on delete cascade,
  ticker text not null,
  shares numeric not null,
  entry_price numeric not null,
  entry_time timestamptz default now(),
  thesis text,
  prediction_id text,
  closed boolean default false,
  exit_price numeric,
  exit_time timestamptz,
  exit_reason text,
  stop_loss_pct numeric,
  take_profit_pct numeric
);
create index if not exists positions_portfolio_open_idx on positions(portfolio_id, closed);

create table if not exists trades (
  id bigserial primary key,
  portfolio_id text not null,
  ticker text not null,
  action text not null check (action in ('buy', 'sell')),
  shares numeric not null,
  price numeric not null,
  fee numeric default 0,
  thesis text,
  prediction_id text,
  created_at timestamptz default now()
);
create index if not exists trades_portfolio_idx on trades(portfolio_id, created_at desc);

create table if not exists predictions (
  id text primary key,
  made_on date not null,
  ticker text not null,
  strategy text not null,
  horizon_days int not null,
  direction text,
  magnitude_pct numeric,
  confidence int,
  rationale text,
  headline text,
  publisher text,
  filing jsonb,
  evidence jsonb,
  price_at_prediction numeric,
  resolves_on date,
  resolved boolean default false,
  resolved_on date,
  actual_pct numeric,
  actual_direction text,
  correct_direction boolean,
  magnitude_error numeric,
  postmortem text,
  created_at timestamptz default now()
);
create index if not exists predictions_made_on_idx on predictions(made_on desc);
create index if not exists predictions_ticker_idx on predictions(ticker);
create index if not exists predictions_resolved_idx on predictions(resolved, resolves_on);
create index if not exists predictions_strategy_idx on predictions(strategy);

create table if not exists reports_cache (
  ticker text primary key,
  payload jsonb not null,
  created_at timestamptz default now()
);

-- seed portfolios
insert into portfolios (id, cash, starting_cash) values
  ('agent', 1000.00, 1000.00),
  ('human', 1000.00, 1000.00)
on conflict (id) do nothing;

-- public read access on portfolios + positions + trades for the dashboard
-- (writes go through service-role key in serverless functions, never browser)
alter table portfolios enable row level security;
alter table positions enable row level security;
alter table trades enable row level security;
alter table predictions enable row level security;
alter table reports_cache enable row level security;

drop policy if exists "public read portfolios" on portfolios;
drop policy if exists "public read positions" on positions;
drop policy if exists "public read trades" on trades;
drop policy if exists "public read predictions" on predictions;
drop policy if exists "public read reports" on reports_cache;

create policy "public read portfolios" on portfolios for select using (true);
create policy "public read positions" on positions for select using (true);
create policy "public read trades" on trades for select using (true);
create policy "public read predictions" on predictions for select using (true);
create policy "public read reports" on reports_cache for select using (true);
