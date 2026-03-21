"""
db.py  —  MySQL connection pool + helpers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared by Live_attendance.py and future scripts.
Uses mysql.connector.pooling for connection reuse.
All queries auto-retry on dropped connection.

Usage:
  import db as DB
  conn = DB.get_conn()
  rows = DB.execute(conn, "SELECT ...", (args,), fetch=True)
  conn.close()   # returns to pool, not actually closed
"""

import sys
import time
import mysql.connector
from mysql.connector import pooling, MySQLConnection
from typing import Any, Optional, Tuple

try:
    from config import db_password
except ImportError:
    db_password = "your_password"


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SCHOOL_ID = 1

_DB_CONFIG = {
    "host":               "localhost",
    "port":               3306,
    "user":               "root",
    "password":           db_password,
    "database":           "attendance_system",
    "autocommit":         False,
    "connection_timeout": 10,
    "use_pure":           True,
}

_POOL_SIZE = 5


# ─────────────────────────────────────────────────────────────────────────────
#  POOL
# ─────────────────────────────────────────────────────────────────────────────

_pool: Optional[pooling.MySQLConnectionPool] = None


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        try:
            _pool = pooling.MySQLConnectionPool(
                pool_name="attendance_pool",
                pool_size=_POOL_SIZE,
                pool_reset_session=True,
                **_DB_CONFIG,
            )
        except mysql.connector.Error as e:
            print(f"\n[DB] Pool creation failed: {e}")
            print(f"  host = {_DB_CONFIG['host']}:{_DB_CONFIG['port']}")
            print(f"  user = {_DB_CONFIG['user']}")
            sys.exit(1)
    return _pool


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_conn() -> MySQLConnection:
    """
    Get a connection from the pool.
    Always call conn.close() when done — returns connection to pool.
    """
    for attempt in range(3):
        try:
            return _get_pool().get_connection()
        except mysql.connector.errors.PoolExhausted:
            time.sleep(0.1 * (attempt + 1))
        except mysql.connector.Error:
            if attempt == 2:
                raise
            time.sleep(0.5)
    raise RuntimeError("Could not get DB connection after 3 attempts")


def execute(conn: MySQLConnection,
            sql: str,
            args: Tuple = (),
            fetch: bool = False) -> Any:
    """
    Execute SQL with auto-reconnect on OperationalError.
    Returns list of dicts if fetch=True, else None.
    Auto-commits for INSERT / UPDATE / DELETE.
    """
    for attempt in range(3):
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, args)
            if fetch:
                result = cur.fetchall()
                cur.close()
                return result
            else:
                conn.commit()
                cur.close()
                return None
        except mysql.connector.errors.OperationalError:
            if attempt < 2:
                try:
                    conn.reconnect(attempts=2, delay=1)
                except Exception:
                    pass
            else:
                raise
        except mysql.connector.IntegrityError:
            raise
        except Exception:
            raise


def ping() -> bool:
    """Returns True if MySQL is reachable."""
    try:
        conn = get_conn()
        conn.ping(reconnect=True)
        conn.close()
        return True
    except Exception:
        return False