-- Run this in Supabase SQL Editor (project ref: mvprtzaatfbvdxwutrbo).
-- Purpose:
-- Backend tables for the financial-literacy pivot — Budgeting, Retirement, and
-- the Credit Behavior Simulator tabs. Each table is one row per user (upsert
-- target on user_id), owned by the authenticated user via RLS.

-- ═══════════════════════════════════════════════════════════════════════════
-- user_budgets
-- ═══════════════════════════════════════════════════════════════════════════

create table if not exists public.user_budgets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  income numeric,
  categories jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

alter table public.user_budgets enable row level security;

drop policy if exists "users can read own budget" on public.user_budgets;
create policy "users can read own budget"
on public.user_budgets
for select
using (auth.uid() = user_id);

drop policy if exists "users can insert own budget" on public.user_budgets;
create policy "users can insert own budget"
on public.user_budgets
for insert
with check (auth.uid() = user_id);

drop policy if exists "users can update own budget" on public.user_budgets;
create policy "users can update own budget"
on public.user_budgets
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- user_retirement_plans
-- (already referenced in the frontend's generated Supabase types, but never
-- actually created/used until now — column names match those types exactly)
-- ═══════════════════════════════════════════════════════════════════════════

create table if not exists public.user_retirement_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  plan_name text,
  current_age int,
  retirement_age int,
  current_savings numeric,
  monthly_contribution numeric,
  expected_return_rate numeric,
  inflation_rate numeric,
  retirement_goal text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

alter table public.user_retirement_plans enable row level security;

drop policy if exists "users can read own retirement plan" on public.user_retirement_plans;
create policy "users can read own retirement plan"
on public.user_retirement_plans
for select
using (auth.uid() = user_id);

drop policy if exists "users can insert own retirement plan" on public.user_retirement_plans;
create policy "users can insert own retirement plan"
on public.user_retirement_plans
for insert
with check (auth.uid() = user_id);

drop policy if exists "users can update own retirement plan" on public.user_retirement_plans;
create policy "users can update own retirement plan"
on public.user_retirement_plans
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- user_credit_sim_profiles
-- ═══════════════════════════════════════════════════════════════════════════

create table if not exists public.user_credit_sim_profiles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  starting_fico int,
  monthly_income numeric,
  monthly_debt numeric,
  behavior_assumptions jsonb,
  last_trajectory jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

alter table public.user_credit_sim_profiles enable row level security;

drop policy if exists "users can read own credit sim profile" on public.user_credit_sim_profiles;
create policy "users can read own credit sim profile"
on public.user_credit_sim_profiles
for select
using (auth.uid() = user_id);

drop policy if exists "users can insert own credit sim profile" on public.user_credit_sim_profiles;
create policy "users can insert own credit sim profile"
on public.user_credit_sim_profiles
for insert
with check (auth.uid() = user_id);

drop policy if exists "users can update own credit sim profile" on public.user_credit_sim_profiles;
create policy "users can update own credit sim profile"
on public.user_credit_sim_profiles
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
