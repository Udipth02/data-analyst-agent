#!/bin/bash

echo "Sending questions.txt to Data Analyst Agent..."

curl -X POST "http://127.0.0.1:8000/api/" \
  -F "file=@questions.txt" \
  -H "accept: application/json" \
  -o response.json

echo -e "\n\n--- Agent Log (output.txt) ---"
cat output.txt

echo -e "\n\n--- API Response (response.json) ---"
cat response.json