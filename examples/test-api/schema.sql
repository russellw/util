CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    file TEXT,
    name text not null,
    icon256 blob,
    text TEXT,
    raw_data blob,
    added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;
