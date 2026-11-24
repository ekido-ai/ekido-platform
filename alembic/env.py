"""Alembic environment for the Platform API: its models are the source of truth, and DATABASE_URL picks the database."""

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from ekido_platform import accounts, payments, plans, privacy, v1_admin, v1_assistant, v1_datasets, v1_invoices  # noqa: F401  (v1_assistant: assistant_teams, assistant_unsupported; v1_datasets: datasets, quotes; v1_admin: admin_actions; payments: orders, events, ledger; plans: team_plans, plan_usage; privacy: deletion_requests, erasure_tasks)

config = context.config
url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("Set DATABASE_URL to the database to migrate.")
config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
# accounts defines its tables on the Platform's Base, so its metadata covers every Platform table.
target_metadata = accounts.Base.metadata


def run_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, render_as_batch=url.startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = engine_from_config(config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as connection:
        # Batch mode lets SQLite (tests, the old server) apply column changes by rebuilding tables.
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=connection.dialect.name == "sqlite", compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
