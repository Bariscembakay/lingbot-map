#!/bin/bash
BASE_URL="https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0"
# BlendedMVS 
for i in $(seq -w 1 15); do
    FILE="BlendedMVS.z$i"
    echo "Downloading $FILE..."
    wget -c "${BASE_URL}/${FILE}"
done

# Download the final .zip descriptor
ZIP_FILE="BlendedMVS.zip"
echo "Downloading $ZIP_FILE..."
wget -c "${BASE_URL}/${ZIP_FILE}"
echo "BlendedMVS downloaded."

# Base URL
# Change to v1.0.2 for BlendedMVS++ 
BASE_URL="https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.1"

# Change to BlendedMVS2 for BlendedMVS++ 
# Download split files BlendedMVS1.z01 to BlendedMVS1.z42
for i in $(seq -w 1 42); do
    FILE="BlendedMVS1.z$i"
    echo "Downloading $FILE..."
    wget -c "${BASE_URL}/${FILE}"
done

# Download the final .zip descriptor
ZIP_FILE="BlendedMVS1.zip"
echo "Downloading $ZIP_FILE..."
wget -c "${BASE_URL}/${ZIP_FILE}"
echo "BlendedMVS+ downloaded."


BASE_URL="https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.2"

# Change to BlendedMVS2 for BlendedMVS++ 
# Download split files BlendedMVS2.z01 to BlendedMVS2.z42
for i in $(seq -w 1 42); do
    FILE="BlendedMVS2.z$i"
    echo "Downloading $FILE..."
    wget -c "${BASE_URL}/${FILE}"
done

# Download the final .zip descriptor
ZIP_FILE="BlendedMVS2.zip"
echo "Downloading $ZIP_FILE..."
wget -c "${BASE_URL}/${ZIP_FILE}"
echo "BlendedMVS++ downloaded."

echo "All files downloaded."

zip -s- BlendedMVS.zip -O combined.zip
unzip combined.zip
rm combined.zip
rm BlendedMVS.z*
mv BlendedMVS/* .
rmdir BlendedMVS

zip -s- BlendedMVS1.zip -O combined1.zip
unzip combined1.zip
rm combined1.zip
rm BlendedMVS1.z*

zip -s- BlendedMVS2.zip -O combined2.zip
unzip combined2.zip
rm combined2.zip
rm BlendedMVS2.z*