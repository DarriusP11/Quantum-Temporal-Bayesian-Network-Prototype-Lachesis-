-- Run this in Supabase SQL Editor (project ref: mvprtzaatfbvdxwutrbo).
-- Purpose:
-- Backend table for the Debt Management Simulator tab (Phase 3) — one row per
-- user (upsert target on user_id), owned by the authenticated user via RLS.
-- Kept as a separate file from the earlier migrations so this one can be run
-- independently.

-- ═══════════════════════════════════════════════════════════════════════════
-- user_debt_plans
-- `debts` stores the full list of debt entries as a jsonb array
-- ({id, name, type, balance, apr_pct, minimum_payment}), since a user can have
-- any number of debts.
-- ═══════════════════════════════════════════════════════════════════════════

create table if not exists public.user_debt_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  debts jsonb,
  extra_monthly_payment numeric,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

alter table public.user_debt_plans enable row level security;

drop policy if exists "users can read own debt plan" on public.user_debt_plans;
create policy "users can read own debt plan"
on public.user_debt_plans
for select
using (auth.uid() = user_id);

drop policy if exists "users can insert own debt plan" on public.user_debt_plans;
create policy "users can insert own debt plan"
on public.user_debt_plans
for insert
with check (auth.uid() = user_id);

drop policy if exists "users can update own debt plan" on public.user_debt_plans;
create policy "users can update own debt plan"
on public.user_debt_plans
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
