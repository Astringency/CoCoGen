"""Redirect historical study reads without editing hash-bound manifests."""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import io
import os
from pathlib import Path
import sys


_active = False


@contextmanager
def mapped_study_reads(root, original_study, blocked_external_paths=()):
    """Process-local mapping for the unchanged training entry point.

    Only old-study reads are redirected. Writes through an old filename and
    unmapped access to the original study are rejected. Python open auditing
    is an application-level check, not an operating-system sandbox.
    """
    global _active
    if not Path(original_study).is_absolute():
        raise ValueError('The historical study root must be absolute')
    root = Path(root).resolve(strict=True)
    original = Path(os.path.abspath(original_study))
    if _active or not root.is_dir() or root == original or root.is_relative_to(original) or original.is_relative_to(root):
        raise ValueError('Use a separate archive root and one mapping context per process')
    blocked = {os.path.abspath(os.fsdecode(p)) for p in blocked_external_paths}
    stats = dict(mapped_opens=0, mapped_files={}, direct_original_access_attempts=[])
    enabled = [True]
    original_open, original_io_open = builtins.open, io.open

    def guard(event, args):
        if not enabled[0] or event != 'open' or not args or isinstance(args[0], int):
            return
        name = os.path.abspath(os.fsdecode(args[0]))
        if Path(name).is_relative_to(original) or name in blocked:
            stats['direct_original_access_attempts'].append(name)
            raise PermissionError('Archive execution may not access original data: '+name)

    def mapped(file, mode):
        if isinstance(file, int):
            return file
        path = Path(os.path.abspath(os.fsdecode(file)))
        if not path.is_relative_to(original):
            return file
        if any(flag in mode for flag in ('w', 'a', 'x', '+')):
            raise PermissionError('Historical study paths are read-only')
        relative = path.relative_to(original)
        target = (root/relative).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Mapped file leaves archive or is not a regular file')
        stats['mapped_opens'] += 1
        stats['mapped_files'][str(relative)] = str(target)
        return target

    def open_builtin(file, mode='r', *args, **kwargs):
        return original_open(mapped(file, mode), mode, *args, **kwargs)

    def open_io(file, mode='r', *args, **kwargs):
        return original_io_open(mapped(file, mode), mode, *args, **kwargs)

    sys.addaudithook(guard)
    _active = True
    builtins.open, io.open = open_builtin, open_io
    try:
        yield stats
    finally:
        builtins.open, io.open = original_open, original_io_open
        enabled[0] = False
        _active = False
