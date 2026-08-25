-- TheMoodShelf — database schema
-- Run this once against a fresh MySQL-compatible database (e.g. TiDB Cloud
-- Serverless) before pointing the app at it. Matches the tables/columns
-- app.py queries, including the columns that used to be added later via
-- ensure_column() (top_emotion, genre, image_url) — folded in here so a
-- fresh database needs no auto-migration step on first run.

CREATE TABLE IF NOT EXISTS users (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(100) NOT NULL,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mood_entries (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT NOT NULL,
    mood            VARCHAR(50) NOT NULL,
    input_text      TEXT NOT NULL,
    top_emotion     VARCHAR(50) DEFAULT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS saved_books (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT NOT NULL,
    title           VARCHAR(500) NOT NULL,
    author          VARCHAR(255),
    mood            VARCHAR(50),
    genre           VARCHAR(100) DEFAULT NULL,
    image_url       VARCHAR(1000) DEFAULT NULL,
    saved_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT NOT NULL,
    content         TEXT NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
