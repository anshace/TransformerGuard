"""
Database Connection Manager for TransformerGuard
Handles SQLAlchemy engine, session creation, and table management
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import yaml
from sqlalchemy import create_engine, engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


class DatabaseConnection:
    """
    Manages database connections and sessions for TransformerGuard.
    Supports SQLite (default) and PostgreSQL databases.
    """

    _instance: Optional["DatabaseConnection"] = None
    _engine: Optional[engine.Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_url: Database URL. If None, loads from config/settings.yaml
        """
        if db_url is None:
            db_url = self._load_db_url_from_config()

        self.db_url = db_url
        self._initialize_engine()

    @classmethod
    def _load_db_url_from_config(cls) -> str:
        """
        Load database URL from config/settings.yaml.

        Returns:
            Database URL string
        """
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            db_config = config.get("database", {})
            db_type = db_config.get("type", "sqlite")

            if db_type == "postgresql":
                # For PostgreSQL, construct from components
                host = db_config.get("host", "localhost")
                port = db_config.get("port", 5432)
                database = db_config.get("database", "transformerguard")
                user = db_config.get("user", "postgres")
                password = db_config.get("password", "")
                return f"postgresql://{user}:{password}@{host}:{port}/{database}"
            else:
                # SQLite - use path from config
                db_path = db_config.get("path", "data/transformerguard.db")
                # Make path absolute if relative
                if not os.path.isabs(db_path):
                    base_dir = Path(__file__).parent.parent.parent
                    db_path = base_dir / db_path
                # Ensure directory exists
                db_path.parent.mkdir(parents=True, exist_ok=True)
                return f"sqlite:///{db_path}"
        except Exception as e:
            # Fallback to default SQLite
            base_dir = Path(__file__).parent.parent.parent
            db_path = base_dir / "data" / "transformerguard.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine and session factory."""
        connect_args = {}

        # SQLite-specific options
        if self.db_url.startswith("sqlite"):
            connect_args = {
                "check_same_thread": False,
                "timeout": 30,
            }

        # Create engine
        self._engine = create_engine(
            self.db_url,
            connect_args=connect_args,
            echo=False,
            pool_pre_ping=True,
        )

        # Create session factory
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> engine.Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            self._initialize_engine()
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get the session factory."""
        if self._session_factory is None:
            self._initialize_engine()
        return self._session_factory

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy Session instance
        """
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.

        Yields:
            SQLAlchemy Session instance

        Usage:
            with db_connection.session_scope() as session:
                session.add(new_object)
                session.commit()
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self._engine)

    def drop_tables(self):
        """Drop all tables (for testing)."""
        Base.metadata.drop_all(bind=self._engine)

    def recreate_tables(self):
        """Drop and recreate all tables."""
        self.drop_tables()
        self.create_tables()

    @classmethod
    def get_instance(cls, db_url: Optional[str] = None) -> "DatabaseConnection":
        """
        Get singleton instance of DatabaseConnection.

        Args:
            db_url: Optional database URL override

        Returns:
            DatabaseConnection instance
        """
        if cls._instance is None:
            cls._instance = cls(db_url)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:
            if cls._instance._engine is not None:
                cls._instance._engine.dispose()
            cls._instance = None
            cls._engine = None
            cls._session_factory = None


# Global instance
_db_connection: Optional[DatabaseConnection] = None


def init_db(db_url: Optional[str] = None) -> DatabaseConnection:
    """
    Initialize the database connection.

    Args:
        db_url: Optional database URL override

    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    _db_connection = DatabaseConnection.get_instance(db_url)
    return _db_connection


def get_session() -> Session:
    """
    Get a database session from the global connection.

    Returns:
        SQLAlchemy Session instance

    Raises:
        RuntimeError: If database not initialized
    """
    global _db_connection

    if _db_connection is None:
        _db_connection = init_db()

    return _db_connection.get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope for database operations.

    Yields:
        SQLAlchemy Session instance

    Usage:
        from src.database import session_scope
        with session_scope() as session:
            session.add(new_object)
            session.commit()
    """
    global _db_connection

    if _db_connection is None:
        _db_connection = init_db()

    with _db_connection.session_scope() as session:
        yield session
