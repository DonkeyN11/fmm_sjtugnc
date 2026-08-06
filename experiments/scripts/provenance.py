"""Shared provenance metadata recorder for experiment outputs.

Usage in any experiment script:
    from provenance import write_provenance
    write_provenance(
        output_dir=args.output_dir,
        script=__file__,
        data_sources=["data/simulation/sigma_10/no_occlusion/no_fault", ...],
        description="Sigma sweep CaMM vs HMM matching",
        parameters={"k": 16, "phmi": 1e-5, "lag_steps": 0},
    )
"""
import json, os, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone


def _hash_file(path: Path, nbytes: int = 8192) -> str:
    """First + last nbytes SHA256 — fast fingerprint for CSV files."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(min(nbytes, size))
            if size > nbytes * 2:
                f.seek(-nbytes, os.SEEK_END)
                tail = f.read(nbytes)
            else:
                tail = b""
        return hashlib.sha256(head + tail).hexdigest()[:16]
    except Exception:
        return "unreadable"


def write_provenance(output_dir, script, data_sources, description="", parameters=None):
    """Write provenance.json to output_dir documenting data lineage."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(script).resolve()
    script_rel = str(script_path.relative_to(Path.cwd())) if str(script_path).startswith(str(Path.cwd())) else str(script_path)

    data_info = {}
    for ds in (data_sources or []):
        dsp = Path(ds)
        if dsp.exists():
            if dsp.is_dir():
                files = {f.name: _hash_file(f) for f in sorted(dsp.glob("*.csv"))[:20]}
            else:
                files = {dsp.name: _hash_file(dsp)}
            data_info[str(ds)] = {"exists": True, "files": files}
        else:
            data_info[str(ds)] = {"exists": False}

    prov = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": description,
        "script": script_rel,
        "data_sources": data_info,
        "parameters": parameters or {},
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
    }

    out = output_dir / "provenance.json"
    with open(out, "w") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"  Provenance: {out}")
