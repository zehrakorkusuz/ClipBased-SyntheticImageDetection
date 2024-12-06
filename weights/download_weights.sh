#!/bin/bash

# This script downloads the weights and checks the MD5 checksum.
# Information taken from the repository owner's answer in issues discussion:
# https://github.com/grip-unina/ClipBased-SyntheticImageDetection/issues/2


# Define the URL and expected MD5 checksum
URL="https://www.grip.unina.it/download/prog/DMimageDetection/weights_clipdet.zip"
EXPECTED_MD5="36685ab9895760d4eea9d42b2a4c464a"
FILE_NAME="weights_clipdet.zip"

# Download the file
echo "Downloading weights from $URL..."
wget -O $FILE_NAME $URL

# Calculate the MD5 checksum of the downloaded file
echo "Calculating MD5 checksum..."
CALCULATED_MD5=$(md5sum $FILE_NAME | awk '{ print $1 }')

# Compare the calculated MD5 checksum with the expected one
if [ "$CALCULATED_MD5" == "$EXPECTED_MD5" ]; then
    echo "MD5 checksum matches: $CALCULATED_MD5"
    echo "Download successful and file is verified."
else
    echo "MD5 checksum does not match!"
    echo "Expected: $EXPECTED_MD5"
    echo "Got: $CALCULATED_MD5"
    echo "Download might be corrupted."
fi