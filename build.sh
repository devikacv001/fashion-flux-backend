#!/usr/bin/env bash
# Build script for Render deployment

set -e

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create uploads directory
mkdir -p uploads

echo "Build completed successfully!"

