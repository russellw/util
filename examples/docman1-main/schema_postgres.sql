CREATE TABLE documents (
    id SERIAL PRIMARY KEY,                 -- Use 'SERIAL' for auto-increment in PostgreSQL
    file TEXT,
    name TEXT NOT NULL,
    icon128 BYTEA,                         -- Use 'BYTEA' for binary data in PostgreSQL
    text TEXT,
    raw_data BYTEA,
    added TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP  -- TIMESTAMP for date and time in PostgreSQL
);
