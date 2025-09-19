import os
import time
import psycopg
from psycopg import OperationalError, InterfaceError
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()
class ResilientPostgresSaver:
    def __init__(self, max_retries=5, retry_delay=0.5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.conn = None
        self.memory = None
        self._connect()
        
    def _create_connection(self):
        """Create a new connection with proper Supabase configuration"""
        conn_str = os.getenv("POSTGRES_SUPABASE")
        if not conn_str:
            raise ValueError("POSTGRES_SUPABASE environment variable not set")
        
        # Ensure SSL mode is enforced for Supabase
        if "sslmode" not in conn_str:
            conn_str += "?sslmode=require"
            
        return psycopg.connect(
            conn_str,
            autocommit=True,
            prepare_threshold=None,  # Disables pipeline mode explicitly
        )

    def _connect(self):
        """Establish connection with retry logic"""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.conn = self._create_connection()
                self.memory = PostgresSaver(self.conn)
                print(f"PostgreSQL connection established (attempt {attempt})")
                return
            except (OperationalError, InterfaceError) as e:
                if attempt < self.max_retries:
                    print(f"Connection failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to connect after {self.max_retries} attempts")
                    raise

    def _ensure_connection(self):
        """Verify and refresh connection if needed"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (OperationalError, InterfaceError):
            print("Connection lost. Reconnecting...")
            self._connect()

    def __getattr__(self, name):
        """Proxy method calls with connection verification"""
        self._ensure_connection()
        
        # Get the attribute from the underlying memory object
        attr = getattr(self.memory, name)
        
        # Wrap callable methods with retry logic
        if callable(attr):
            def wrapper(*args, **kwargs):
                for attempt in range(1, self.max_retries + 1):
                    try:
                        # Refresh attribute reference after reconnection
                        current_attr = getattr(self.memory, name)
                        return current_attr(*args, **kwargs)
                    except (OperationalError, InterfaceError) as e:
                        if attempt < self.max_retries:
                            print(f"Call to {name}() failed: {e}. Retrying in {self.retry_delay}s...")
                            self._connect()
                            time.sleep(self.retry_delay)
                        else:
                            print(f"Operation {name}() failed after {self.max_retries} retries")
                            raise
            return wrapper
        
        # Return non-callable attributes directly
        return attr
    def close(self):
        """Explicitly close the saver and its connection."""
        try:
            if hasattr(self.memory, "close"):
                self.memory.close()
        finally:
            if self.conn:
                self.conn.close()