#!/bin/bash
# Auto-bump agent patch version when agent source changes.
# Called by Claude Code hook after Edit/Write on packages/agent/src/**
set -e

PKG="packages/agent/package.json"
if [ ! -f "$PKG" ]; then exit 0; fi

# Read current version
VERSION=$(node -e "console.log(require('./$PKG').version)")
IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

# Update package.json
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('$PKG', 'utf-8'));
pkg.version = '$NEW_VERSION';
fs.writeFileSync('$PKG', JSON.stringify(pkg, null, 2) + '\n');
"

echo "Agent version bumped: $VERSION → $NEW_VERSION"
