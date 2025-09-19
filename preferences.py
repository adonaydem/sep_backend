import psycopg as pg
from typing import List, Optional, Dict
import os

DB_URL = os.getenv('POSTGRES_SUPABASE')  # e.g. 'postgresql://user:pass@host:port/db'


def save_preferences(
    user_id: str,
    languages: List[str],
    objects: List[str],
    voice_speed: float,
) -> None:
    """Insert or update user preferences."""
    with pg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_preferences (user_id, languages, objects, voice_speed)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                  languages = EXCLUDED.languages,
                  objects = EXCLUDED.objects,
                  voice_speed = EXCLUDED.voice_speed,
                  updated_at = NOW();
                """,
                (user_id, languages, objects, voice_speed)
            )


def get_preferences(user_id: str) -> Optional[Dict[str, any]]:
    """Fetch preferences for a given user. Returns None if not found."""
    with pg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT languages, objects, voice_speed
                 FROM user_preferences
                 WHERE user_id = %s;""",
                (user_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                'languages': row[0],
                'objects': row[1],
                'voice_speed': row[2],
            }
