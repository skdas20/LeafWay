#!/bin/bash

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Print Python and pip versions for debugging
python --version
pip --version

echo "Build completed successfully!" 