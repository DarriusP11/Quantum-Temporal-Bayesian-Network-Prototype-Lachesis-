-- Run this in Supabase SQL Editor.
-- Purpose:
-- 1) Keep auth credentials in Supabase Auth (auth.users)
-- 2) Mirror app profile data (username, email) into public.profiles for analytics/admin tools (e.g., Chat2DB)

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text unique,
  username text unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email, username)
  values (
    new.id,
    new.email,
    nullif(trim(new.raw_user_meta_data ->> 'username'), '')
  )
  on conflict (id) do update set
    email = excluded.email,
    username = coalesce(excluded.username, public.profiles.username),
    updated_at = now();

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute procedure public.handle_new_user();

alter table public.profiles enable row level security;

drop policy if exists "users can read own profile" on public.profiles;
create policy "users can read own profile"
on public.profiles
for select
using (auth.uid() = id);

drop policy if exists "users can update own profile" on public.profiles;
create policy "users can update own profile"
on public.profiles
for update
using (auth.uid() = id)
with check (auth.uid() = id);

-- Optional: backfill existing auth users that were created before this trigger.
insert into public.profiles (id, email, username)
select
  u.id,
  u.email,
  nullif(trim(u.raw_user_meta_data ->> 'username'), '')
from auth.users u
left join public.profiles p on p.id = u.id
where p.id is null;
