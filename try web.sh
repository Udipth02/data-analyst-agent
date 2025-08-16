#!/bin/bash

echo "Sending questions.txt to Data Analyst Agent..."

curl -X POST "https://data-analyst-agent-jbse.onrender.com/api/" \
  -F "file=@questions.txt" \
  -H "accept: application/json" 
  -o output.txt

echo -e "\n\n--- Output.txt ---"
cat output.txt