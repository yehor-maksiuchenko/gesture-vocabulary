import sqlite3
import json
import time

class GestureDB:
    def __init__(self, path='gestures.db'):
        self.conn = sqlite3.connect(path)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS static_gestures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                vector TEXT NOT NULL,   -- 63 normalized floats as JSON
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS dynamic_gestures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                sequence TEXT NOT NULL, -- list of vectors (frames × 63 floats)
                frame_count INTEGER,
                created_at REAL
            );
        ''')
        self.conn.commit()

    def save_static(self, name, vector):
        self.conn.execute(
            'INSERT INTO static_gestures (name, vector, created_at) VALUES (?,?,?)',
            (name, json.dumps(vector), time.time())
        )
        self.conn.commit()

    def save_dynamic(self, name, sequence):
        self.conn.execute(
            'INSERT INTO dynamic_gestures (name, sequence, frame_count, created_at) VALUES (?,?,?,?)',
            (name, json.dumps(sequence), len(sequence), time.time())
        )
        self.conn.commit()

    def get_all_static(self):
        rows = self.conn.execute('SELECT name, vector FROM static_gestures').fetchall()
        return [(name, json.loads(vec)) for name, vec in rows]

    def get_all_dynamic(self):
        rows = self.conn.execute('SELECT name, sequence FROM dynamic_gestures').fetchall()
        return [(name, json.loads(seq)) for name, seq in rows]