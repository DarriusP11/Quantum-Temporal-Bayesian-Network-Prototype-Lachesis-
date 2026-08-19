-- Run this in Supabase SQL Editor (project ref: mvprtzaatfbvdxwutrbo).
-- Purpose:
-- Adds a webhook-idempotency column to the existing `profiles` table for the
-- single-tier Stripe redesign. Additive only — existing rows get NULL, which
-- the webhook handler treats as "never processed an event for this user yet."
--
-- Assumes `profiles` already has the billing columns described in
-- CONTRACTOR_TOUCHPOINT.md (stripe_customer_id, subscription_id, plan,
-- subscription_status, current_period_end) — if those were never actually
-- added, add them first:
--   alter table public.profiles
--     add column if not exists stripe_customer_id text,
--     add column if not exists subscription_id text,
--     add column if not exists plan text default 'free',
--     add column if not exists subscription_status text,
--     add column if not exists current_period_end timestamptz;

alter table public.profiles
  add column if not exists last_webhook_event_id text;
