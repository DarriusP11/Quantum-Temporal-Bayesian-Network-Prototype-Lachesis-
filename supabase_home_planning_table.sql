-- Run this in Supabase SQL Editor (project ref: mvprtzaatfbvdxwutrbo).
-- Purpose:
-- Backend table for the Home Planning tab (Phase 2) — one row per user
-- (upsert target on user_id), owned by the authenticated user via RLS.
-- Kept as a separate file from supabase_financial_literacy_tables.sql so this
-- migration can be run independently without re-running the earlier one.

-- ═══════════════════════════════════════════════════════════════════════════
-- user_home_plans
-- option_a / option_b each store {type, inputs} for one of the four housing
-- types (home, apartment, dorm, mobile_home) — kept as jsonb so the schema
-- doesn't need a column per housing type.
-- ═══════════════════════════════════════════════════════════════════════════

create table if not exists public.user_home_plans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  option_a jsonb,
  option_b jsonb,
  utilities jsonb,
  comparison_settings jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id)
);

alter table public.user_home_plans enable row level security;

drop policy if exists "users can read own home plan" on public.user_home_plans;
create policy "users can read own home plan"
on public.user_home_plans
for select
using (auth.uid() = user_id);

drop policy if exists "users can insert own home plan" on public.user_home_plans;
create policy "users can insert own home plan"
on public.user_home_plans
for insert
with check (auth.uid() = user_id);

drop policy if exists "users can update own home plan" on public.user_home_plans;
create policy "users can update own home plan"
on public.user_home_plans
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);
