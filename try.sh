#!/bin/bash

echo "Sending questions.txt to Data Analyst Agent..."

curl -X POST "http://127.0.0.1:8000/api/" \
  -F "file=@questions.txt" \
  -H "accept: application/json"

echo -e "\n\n--- Output.txt ---"
cat output.txt