#!/usr/bin/env bash
# ┌──────────────────────────────────────────────────────────────────────┐
# │         cleanup_and_push.sh « Fix & Finalize »                      │
# │                                                                      │
# │  Run from inside ~/PyProjects/reRandomStats/                        │
# │  Removes the broken brace-expansion directory, cleans up legacy     │
# │  files, fixes the git remote, and pushes.                           │
# └──────────────────────────────────────────────────────────────────────┘

set -euo pipefail

REMOTE="git@github.com:zerotonin/rerandomstats.git"

echo "══════════════════════════════════════════════════════════════"
echo "  reRandomStats — Cleanup & Push"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Remove the broken brace-expansion directory ───────────────────
echo ">>> Removing broken literal brace directory..."
rm -rf '{rerandomstats,tests,docs'

# Verify it's gone
if [ -d '{rerandomstats,tests,docs' ]; then
    echo "ERROR: Could not remove broken directory. Try manually:"
    echo "  rm -rf '{rerandomstats,tests,docs'"
    exit 1
fi
echo "    Done."

# ── 2. Remove old root-level Python files (now in rerandomstats/) ────
echo ">>> Removing legacy root-level .py files..."
LEGACY_FILES=(
    binominalStats.py
    binominalStats
    dataIO.py
    FisherExact.py
    FisherResampling.py
    HypothesisTests.py
    multiGroupTest.py
    resampleNofK.py
    run_all_tests.py
    variance_ana_Fig1F.py
    write_pretty_table.py
    reRandomStats.yaml
    LICENSE.txt
    setup_repo.sh
)

for f in "${LEGACY_FILES[@]}"; do
    if [ -e "$f" ]; then
        rm -f "$f"
        echo "    Removed: $f"
    fi
done

# ── 3. Remove __pycache__ ────────────────────────────────────────────
echo ">>> Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "    Done."

# ── 4. Stage the deletions ───────────────────────────────────────────
echo ">>> Staging cleanup commit..."
git add -A
git commit -m "chore: remove legacy root-level files and broken directory

- Deleted original .py files (now refactored in rerandomstats/)
- Removed broken brace-expansion directory artifact
- Cleaned up __pycache__ and duplicate LICENSE.txt
- Legacy data/ and stats/ directories preserved for branch history"

# ── 5. Fix the remote and push ───────────────────────────────────────
echo ">>> Fixing git remote..."
if git remote get-url origin &>/dev/null; then
    git remote set-url origin "$REMOTE"
    echo "    Updated existing remote to: $REMOTE"
else
    git remote add origin "$REMOTE"
    echo "    Added remote: $REMOTE"
fi

echo ">>> Pushing to GitHub..."
#git push -u origin main

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done! Repo live at: https://github.com/zerotonin/rerandomstats"
echo ""
echo "  Next steps for Zenodo DOI:"
echo "    1. https://zenodo.org/account/settings/github/"
echo "       -> Enable zerotonin/rerandomstats"
echo "    2. Tag and push:"
echo ""
echo "       git tag -a v0.1.0 -m 'v0.1.0: initial release'"
echo "       git push origin v0.1.0"
echo ""
echo "    3. release.yml auto-creates the GitHub Release"
echo "    4. Zenodo mints the DOI from CITATION.cff"
echo ""
echo "  GitHub Pages:"
echo "    Settings -> Pages -> Source: GitHub Actions"
echo "══════════════════════════════════════════════════════════════"
