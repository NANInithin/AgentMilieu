CREATE TABLE IF NOT EXISTS accounts (
    id uuid PRIMARY KEY,
    google_sub text UNIQUE NOT NULL,
    email text NOT NULL,
    name text NOT NULL,
    balance_mills bigint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash text PRIMARY KEY,
    account_id uuid NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS desktop_logins (
    id uuid PRIMARY KEY,
    secret_hash text NOT NULL,
    code text NOT NULL,
    account_id uuid REFERENCES accounts(id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS topups (
    id uuid PRIMARY KEY,
    account_id uuid NOT NULL REFERENCES accounts(id),
    amount_mills integer NOT NULL CHECK (amount_mills IN (5000, 10000, 20000)),
    mollie_id text UNIQUE,
    status text NOT NULL DEFAULT 'creating',
    checkout_url text,
    credited_mills integer NOT NULL DEFAULT 0,
    reversed_mills integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE topups ADD COLUMN IF NOT EXISTS mollie_snapshot_at timestamptz;

CREATE TABLE IF NOT EXISTS generations (
    account_id uuid NOT NULL REFERENCES accounts(id),
    id uuid NOT NULL,
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    error text,
    pack_id text,
    lease_until timestamptz NOT NULL DEFAULT now() + interval '15 minutes',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, id)
);

CREATE UNIQUE INDEX IF NOT EXISTS generations_one_active
    ON generations (account_id) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS packs (
    id text PRIMARY KEY,
    account_id uuid NOT NULL REFERENCES accounts(id),
    payload jsonb NOT NULL,
    pack_hash text NOT NULL,
    approved_hash text,
    status text NOT NULL DEFAULT 'Pending review'
        CHECK (status IN ('Pending review', 'Approved', 'Rejected')),
    notion_page_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS wallet_entries (
    account_id uuid NOT NULL REFERENCES accounts(id),
    kind text NOT NULL,
    reference text NOT NULL,
    amount_mills integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, kind, reference)
);

CREATE INDEX IF NOT EXISTS wallet_entries_by_account ON wallet_entries (account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS packs_by_account ON packs (account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS topups_by_account ON topups (account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS sessions_expiration ON sessions (expires_at);
CREATE INDEX IF NOT EXISTS desktop_logins_expiration ON desktop_logins (expires_at);
