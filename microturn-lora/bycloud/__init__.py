"""bycloud deep pass: catalogue, captions/transcripts and extraction prompts.

The caption path is research-ingest's (yt-dlp ``--skip-download --write-subs
--write-auto-subs``, English, vtt); this package adds the per-video transcript
schema the bycloud task needs and the rolling auto-caption de-duplication.

Raw data stays under ``data/private/bycloud/`` (gitignored):
``flat.jsonl`` (yt-dlp flat playlist), ``catalogue.json``, ``videos.json``,
``raw/`` (vtt + info.json), ``transcripts/<id>.json``, ``no_captions.json``.
"""

from .ingest import (  # noqa: F401
    CATALOGUE,
    PRIVATE,
    TRANSCRIPTS,
    VIDEOS,
    build_catalogue,
    load_videos,
    parse_vtt,
    select_videos,
    write_transcript,
)
