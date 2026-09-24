#!/usr/bin/env bash
set -e

REPO="https://github.com/SuperSonicPro/RetroMan.git"
USERNAME="TheGeek2001"

echo "=== RetroMan GitHub Uploader ==="
echo

# Git author information
git config user.name "SuperSonicPro"

# Initialize repository if necessary
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git branch -M main
fi

# Make sure we're pointing at the correct RetroMan repository
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REPO"
else
    git remote add origin "$REPO"
fi

echo "Adding RetroMan source files..."
git add -A

# Commit only if something changed
if ! git diff --cached --quiet; then
    git commit -m "Update RetroMan source"
else
    echo "No new local changes to commit."
fi

# Incorporate anything already on GitHub
echo "Synchronizing with GitHub..."
git pull origin main --rebase || {
    echo
    echo "Automatic rebase failed."
    echo "Resolve any conflicts, then run this script again."
    exit 1
}

echo
echo "Uploading to:"
echo "$REPO"
echo
echo "GitHub username: $USERNAME"
echo "When authentication is requested, use a Personal Access Token,"
echo "NOT your GitHub password."
echo

git push -u origin main

echo
echo "RetroMan upload complete."
