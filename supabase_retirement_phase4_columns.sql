-- Run this in Supabase SQL Editor (project ref: mvprtzaatfbvdxwutrbo).
-- Purpose:
-- Adds the three Phase 4 columns to the existing user_retirement_plans table
-- (created by supabase_financial_literacy_tables.sql) for the new withdrawal
-- / decumulation risk analysis. Additive only — existing rows are backfilled
-- with the DEFAULT values shown below, so nothing in the already-working
-- accumulation calculator breaks once this runs.
--
-- IMPORTANT: run this BEFORE using the updated Retirement tab. The backend's
-- retirement.py now always sends these three fields when saving a plan, so
-- until this migration runs, saving a retirement plan (even just the basic
-- accumulation numbers) will fail with a "column not found" error from
-- PostgREST. If you haven't yet run supabase_financial_literacy_tables.sql
-- (Phase 1) at all, run that one first — this file only adds columns to a
-- table it assumes already exists.

alter table public.user_retirement_plans
  add column if not exists roth_pct numeric default 0,
  add column if not exists life_expectancy int default 90,
  add column if not exists withdrawal_rate_pct numeric default 4.0;
