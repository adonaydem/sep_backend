from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
import os

load_dotenv()

pool = ConnectionPool(conninfo=os.getenv('POSTGRES_SUPABASE')) 

def insert_voice(from_name: str,from_uid: str, to_uid: str, filename: str, read: bool = False) -> int | None:

    new_id = None

    # Acquire a connection from the pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                # Insert the row; `id` is bigserial, `created_at` defaults to now()
                cur.execute(
                    """
                    INSERT INTO voice_chat (from_uid, from_name,to_uid, filename, read)
                    VALUES (%s, %s,%s, %s, %s)
                    RETURNING id;
                    """,
                    (from_uid, from_name,to_uid, filename, read),
                )
                new_id = cur.fetchone()[0]
                # When exiting the `with conn:` block, psycopg3 will automatically commit 
                # (because autocommit=False). If an exception occurs, it will rollback.
                print(f"__Inserted voice record with id = {new_id}")
            except Exception as e:
                # Any exception here will cause a rollback at context exit
                print(f"__Error during INSERT: {e}")
                return {"Error": str(e)}
                new_id = None

    return {"id":new_id}

def get_unread_by_to_uid(to_uid: str) -> list[dict]:
    """
    Returns a list of dicts, each representing one unread voice record
    for the specified `to_uid`.
    """
    
    results = []

    with pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT id, from_uid, from_name,filename, created_at
                    FROM voice_chat
                    WHERE to_uid = %s
                      AND read = FALSE
                    ORDER BY created_at desc;
                    """,
                    (to_uid,),
                )
                rows = cur.fetchall()
                for row in rows:
                    # row = (id, from_uid, filename, created_at)
                    record = {
                        "id": row[0],
                        "from_uid": row[1],
                        "from_names": row[2],
                        "filename": row[3],
                        "created_at": row[4],
                    }
                    results.append(record)
            except Exception as e:
                print(f"[!] Error during SELECT: {e}")
                # No rollback needed for a SELECT, but it will “exit” the transaction block if used.
    return results

def get_chat_history(to_uid: str, from_uid: str) -> list[dict]:
    """
    Returns a list of dicts, each representing one unread voice record
    for the specified `to_uid`.
    """
    
    results = []
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT id, from_uid, from_name,to_uid,filename, created_at
                        FROM voice_chat
                        WHERE (to_uid = %s
                        AND from_uid = %s) OR (from_uid = %s AND to_uid = %s)
                        ORDER BY created_at desc;
                        """,
                        (to_uid, from_uid,to_uid, from_uid),
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        # row = (id, from_uid, filename, created_at)
                        record = {
                            "id": str(row[0]),
                            "from_uid": row[1],
                            "from_name": row[2],
                            "to_uid": row[3],
                            "filename": row[4],
                            "created_at": str(row[5]),
                        }
                        print(record)
                        results.append(record)
                except Exception as e:
                    print(f"[!] Error during SELECT: {e}")
                    # No rollback needed for a SELECT, but it will “exit” the transaction block if used.
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
    
    return results


def mark_read(id):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE voice_chat
                    SET read = TRUE
                    WHERE id = %s;
                    """,
                    (id,),
                )
    except Exception as e:
        print(f"[!] Error during UPDATE: {e}")


def get_chat_list(to_uid):
    try:
        results = []
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (from_uid) id, from_uid, to_uid,from_name, created_at
                    FROM voice_chat
                    WHERE to_uid = %s
                    ORDER BY from_uid, created_at DESC;
                    """,
                    (to_uid,)
                )
                rows = cur.fetchall()
                rows.sort(key=lambda r: r[4], reverse=True)
                results = []
                for row in rows:
                    results.append(
                        {
                            "id": row[0],
                            "from_uid": row[1],
                            "to_uid": row[2],
                            "from_name": row[3],
                            "created_at": row[4],
                        }
                    )

                return results
    except Exception as e:
        print(f"[!] Error during SELECT: {e}")
        return []
    