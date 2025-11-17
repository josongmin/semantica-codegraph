"""초기 스키마 생성

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector 확장 (필요한 경우)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # code_nodes 테이블 생성
    op.create_table(
        "code_nodes",
        sa.Column("repo_id", sa.Text(), nullable=False),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("span_start_line", sa.Integer(), nullable=False),
        sa.Column("span_start_col", sa.Integer(), nullable=False),
        sa.Column("span_end_line", sa.Integer(), nullable=False),
        sa.Column("span_end_col", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("attrs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("repo_id", "id"),
    )

    # code_edges 테이블 생성
    op.create_table(
        "code_edges",
        sa.Column("repo_id", sa.Text(), nullable=False),
        sa.Column("src_id", sa.Text(), nullable=False),
        sa.Column("dst_id", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("attrs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("repo_id", "src_id", "dst_id", "type"),
    )

    # 인덱스 생성
    op.create_index(
        "idx_nodes_file_path",
        "code_nodes",
        ["repo_id", "file_path"],
    )

    op.create_index(
        "idx_nodes_location",
        "code_nodes",
        ["repo_id", "file_path", "span_start_line", "span_end_line"],
    )

    op.create_index(
        "idx_nodes_name",
        "code_nodes",
        ["repo_id", "name"],
    )

    op.create_index(
        "idx_edges_src",
        "code_edges",
        ["repo_id", "src_id"],
    )

    op.create_index(
        "idx_edges_dst",
        "code_edges",
        ["repo_id", "dst_id"],
    )


def downgrade() -> None:
    # 인덱스 삭제
    op.drop_index("idx_edges_dst", table_name="code_edges")
    op.drop_index("idx_edges_src", table_name="code_edges")
    op.drop_index("idx_nodes_name", table_name="code_nodes")
    op.drop_index("idx_nodes_location", table_name="code_nodes")
    op.drop_index("idx_nodes_file_path", table_name="code_nodes")

    # 테이블 삭제
    op.drop_table("code_edges")
    op.drop_table("code_nodes")

