"""MinerU engine adapter (implements ExternalParserBase hooks)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import MINERU_RAW_DIR_SUFFIX, PARSER_ENGINE_MINERU
from lightrag.parser.external._base import ExternalParserBase

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class MinerUParser(ExternalParserBase):
    engine_name = PARSER_ENGINE_MINERU
    raw_dir_suffix = MINERU_RAW_DIR_SUFFIX
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_MINERU"

    def is_bundle_valid(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> bool:
        from lightrag.parser.external.mineru import is_bundle_valid

        return is_bundle_valid(raw_dir, source_path, overrides=engine_params)

    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        upload_name: str,
        engine_params: "Mapping[str, Any] | None" = None,
        runtime_options: "Mapping[str, Any] | None" = None,
    ) -> None:
        from lightrag.parser.external.mineru import MinerURawClient

        await MinerURawClient(
            overrides=engine_params,
            runtime_options=runtime_options,
        ).download_into(raw_dir, source_path, upload_name=upload_name)

    @asynccontextmanager
    async def acquire_remote_parse_lease(
        self,
        ctx,
        source_path: Path,
        *,
        engine_params: "Mapping[str, Any] | None" = None,
    ):
        del engine_params
        from lightrag.parser.external.mineru.scheduling import mineru_parse_lease

        workspace = str(getattr(ctx.rag, "workspace", "") or "default")
        static_request_cap = getattr(ctx.rag, "max_parallel_parse_mineru", None)
        async with mineru_parse_lease(
            workspace=workspace,
            doc_id=ctx.doc_id,
            source_path=source_path,
            static_request_cap=static_request_cap,
        ) as lease:
            yield lease.client_options()

    def build_ir(self, raw_dir: Path, document_name: str) -> "IRDoc":
        from lightrag.parser.external.mineru import MinerUIRBuilder

        return MinerUIRBuilder().normalize_from_workdir(
            raw_dir, document_name=document_name
        )
