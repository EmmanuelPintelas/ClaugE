"""Check and unpack the downloaded parts of the release into the repository.

Download any subset of the release into one folder (sub-folders are fine), together with
SHA256SUMS.txt, then from the repository root:

    python tools/assemble_release.py --downloads /path/to/downloads
    python tools/assemble_release.py --downloads D:/claugE_release --no-verify --keep

Every part is a zip named <name>.<predictions|models|logs>.<i>of<n>.zip whose entries are
paths under methods/ (for example methods/ClaugE/outputs/kather_efficientnet_b0_kmax5/...),
so it unpacks to the place the code reads.  Each part is checked against SHA256SUMS.txt
when it is listed there.  At the end the script lists, per method, the committees whose
predictions are complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PART = re.compile(r"^(?P<name>.+)\.(?P<kind>predictions|models|logs)\.(?P<i>\d+)of(?P<n>\d+)\.zip$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def committee_complete(path):
    """Whether every member's predictions of a committee are on disk."""
    try:
        with open(os.path.join(path, "committee.json"), encoding="utf-8") as f:
            n = len(json.load(f)["members"])
        with open(os.path.join(path, "predictions", "index.json"), encoding="utf-8") as f:
            splits = json.load(f)["splits"]
    except (OSError, KeyError, ValueError):
        return False
    return all(os.path.exists(os.path.join(path, "predictions", s, f"member_{j:04d}.npy"))
               for s in splits for j in range(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downloads", required=True, help="folder holding the downloaded parts")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    ap.add_argument("--keep", action="store_true", help="keep the zip parts after unpacking")
    a = ap.parse_args(argv)

    sums = {}
    for root, _, files in os.walk(a.downloads):
        if "SHA256SUMS.txt" in files:
            with open(os.path.join(root, "SHA256SUMS.txt"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        digest, name = line.split(maxsplit=1)
                        sums[os.path.basename(name.strip())] = digest
    parts = [os.path.join(r, f) for r, _, fs in os.walk(a.downloads) for f in sorted(fs) if PART.match(f)]
    if not parts:
        print(f"no parts found under {a.downloads}")
        return 1
    bad = []
    for path in parts:
        name = os.path.basename(path)
        if not a.no_verify:
            if name not in sums:
                print(f"  {name}: not in SHA256SUMS.txt, unpacked unverified")
            elif sha256(path) != sums[name]:
                print(f"  {name}: CHECKSUM MISMATCH, download it again")
                bad.append(name)
                continue
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                parts_ = member.split("/")
                if parts_[0] != "methods" or ".." in parts_ or member.startswith("/"):
                    raise SystemExit(f"{name}: unexpected entry {member!r}")
            z.extractall(ROOT)
        print(f"  unpacked {name}")
        if not a.keep:
            os.remove(path)

    print("\ncommittees with complete predictions:")
    mroot = os.path.join(ROOT, "methods")
    for method in sorted(os.listdir(mroot)):
        out = os.path.join(mroot, method, "outputs")
        if os.path.isdir(out):
            done = [d for d in sorted(os.listdir(out)) if committee_complete(os.path.join(out, d))]
            print(f"  {method:15s} {len(done):3d}  {' '.join(done)}")
    if bad:
        print(f"\n{len(bad)} part(s) failed the checksum: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
