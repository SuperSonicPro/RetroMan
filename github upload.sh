#!/usr/bin/env bash
set -e

REPO="https://github.com/SuperSonicPro/RetroMan.git"
ACCOUNT="TheGeek2001"
BRANCH="main"

echo "=== RetroMan Automatic GitHub Uploader ==="

# Make sure we're actually inside the RetroMan source directory.
if [ ! -f "README.md" ] && [ ! -d ".git" ]; then
    echo "WARNING: This does not look like the RetroMan source directory."
    echo "Run this script from inside the extracted RetroMan folder."
    exit 1
fi

# GitHub CLI
if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI is not installed."
    echo
    echo "On Debian/Ubuntu-based Linux:"
    echo "  sudo apt install gh"
    exit 1
fi

# Configure commit identity if missing.
if ! git config --global user.name >/dev/null 2>&1; then
    git config --global user.name "$ACCOUNT"
fi

if ! git config --global user.email >/dev/null 2>&1; then
    echo
    read -r -p "GitHub email address: " EMAIL
    git config --global user.email "$EMAIL"
fi

# Authenticate with GitHub.
echo
echo "Checking GitHub authentication..."

if ! gh auth status >/dev/null 2>&1; then
    echo "GitHub authorization is required."
    echo "A browser login will be started."
    gh auth login --hostname github.com --git-protocol https --web
fi

# Explicitly switch to the desired account when it is available.
if gh auth switch --hostname github.com --user "$ACCOUNT" >/dev/null 2>&1; then
    echo "Using GitHub account: $ACCOUNT"
fi

# Make Git use GitHub CLI authentication instead of stale credentials.
gh auth setup-git

# Initialize Git if necessary.
if [ ! -d ".git" ]; then
    echo "Initializing repository..."
    git init
fi

git branch -M "$BRANCH"

# Repair/create origin.
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REPO"
else
    git remote add origin "$REPO"
fi

echo
echo "Repository:"
git remote get-url origin

# Add everything recursively.
echo
echo "Adding RetroMan files..."
git add -A

# Commit only when necessary.
if ! git diff --cached --quiet; then
    git commit -m "Update RetroMan source"
else
    echo "Local files are already committed."
fi

# Fetch existing GitHub repository.
echo
echo "Synchronizing with GitHub..."
git fetch origin "$BRANCH"

# Merge remote history when necessary.
if git rev-parse "origin/$BRANCH" >/dev/null 2>&1; then
    git merge "origin/$BRANCH" --allow-unrelated-histories --no-edit || {
        echo
        echo "A merge conflict occurred."
        echo "Git cannot safely decide which version you want."
        echo "Resolve the listed files and run this script again."
        exit 1
    }
fi

# Push.
echo
echo "Uploading RetroMan..."
git push -u origin "$BRANCH"

echo
echo "======================================"
echo " RetroMan upload completed successfully"
echo "======================================"
echo
echo "https://github.com/SuperSonicPro/RetroMan"
