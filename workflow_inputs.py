"""Shared, read-only defaults for the two extraction programs."""
import json
import os
import shlex
from string import Template
from pathlib import Path


def read_settings(path):
    """Read saved scalar assignments without executing shell code."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == "declare":
            words = words[2:]
        elif words and words[0] == "export":
            words = words[1:]
        if len(words) == 1 and "=" in words[0]:
            name, value = words[0].split("=", 1)
            if name.isidentifier():
                value = Template(value).safe_substitute({**os.environ, **values})
                if "$" not in value:
                    values[name] = value
    return values


def resolve_inputs(args, parser, sources=False):
    """Fill omitted arguments from the workflow; explicit arguments always win."""
    if args.workflow:
        root = args.workflow.expanduser().resolve()
        if not root.is_dir():
            parser.error(f"Workflow folder not found: {root}")
        args.workflow = root
        saved = read_settings(root / "config.sh")
        saved.update(read_settings(root / "dataset.sh"))
        report_path = root / "input-check.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        obs = args.obs_id or saved.get("OBS_ID")
        analysis = read_settings(Path(__file__).parent / "configs" / f"{obs}.sh") if obs else {}
        snapshots = root / "snapshots" if (root / "snapshots").is_dir() else root
        full = root / "full"
        link = snapshots / "full-workflow.txt"
        if not full.is_dir() and link.is_file():
            full = Path(link.read_text().strip()).expanduser()
            if not full.is_absolute():
                full = snapshots / full
        output = Path(saved["RESULTS_DIR"]).expanduser() if saved.get("RESULTS_DIR") else snapshots / f"plots-{obs}"
        defaults = {"obs_id": obs, "field": saved.get("FIELD"), "field_ra": saved.get("FIELD_RA") or report.get("field_centre", {}).get("ra"), "field_dec": saved.get("FIELD_DEC") or report.get("field_centre", {}).get("dec"), "full": full, "snapshots": snapshots, "output": output, "pointing": output / f"pointing-{obs}.npz" if obs else None, "racs": analysis.get("RACS_FILE") or saved.get("RACS_FILE") or report.get("reference_catalogue")}
        for name, value in defaults.items():
            if hasattr(args, name) and getattr(args, name) is None and value is not None:
                setattr(args, name, value)

    required = ("obs_id", "field", "snapshots", "pointing", "racs", "output") if sources else ("obs_id", "full", "snapshots", "pointing", "racs")
    missing = ["--" + name.replace("_", "-") for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Could not infer " + ", ".join(missing) + ". Supply these explicitly, or use a workflow with saved config.sh and dataset.sh.")
    for name in ("full", "snapshots", "pointing", "racs", "output"):
        if hasattr(args, name) and getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    if not sources and args.output is None:
        args.output = args.snapshots / f"plots-{args.obs_id}"
    if sources and args.last_bin is None:
        windows = args.snapshots / "windows.txt"
        if not windows.is_file():
            parser.error(f"Time-window file not found: {windows}")
        numbers = [int(line.split()[0].removeprefix("bin-")) for line in windows.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not numbers:
            parser.error(f"No time bins in {windows}")
        args.last_bin = max(numbers)
    if args.show_inputs:
        names = required + (("last_bin", "reference_bin", "min_reference_mjy", "source_search_arcsec") if sources else ("field_ra", "field_dec"))
        print(json.dumps({name: str(getattr(args, name)) if isinstance(getattr(args, name), Path) else getattr(args, name) for name in names + (() if "output" in names else ("output",))}, indent=2))
        parser.exit()
    return args
