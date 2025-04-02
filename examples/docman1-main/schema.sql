CREATE TABLE folders (
    id INTEGER PRIMARY KEY,
    name text not null,
    added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;
insert into folders(name) values('Inbox');

CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    name text not null,
    file TEXT,
    text TEXT,
    icon256 blob,
    raw_data blob,
    folder INTEGER NOT NULL,
    added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (folder) REFERENCES folders (id)
) STRICT;
