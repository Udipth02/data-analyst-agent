# prompt_tools.py

import textwrap
import datetime
import json
import os
import re

class PromptBuilder:
    def __init__(self, system_instruction=None):
        self.sections = {}
        self.system_instruction = system_instruction or ""
        self.timestamp = datetime.datetime.now().isoformat()

    def add_section(self, title, content):
        self.sections[title.strip()] = textwrap.dedent(content).strip()

    def build(self):
        prompt_parts = []
        if self.system_instruction:
            prompt_parts.append(f"### System Instruction\n{self.system_instruction.strip()}")
        for title, content in self.sections.items():
            prompt_parts.append(f"### {title}\n{content}")
        return "\n\n".join(prompt_parts)

    def export(self):
        return {
            "timestamp": self.timestamp,
            "prompt": self.build(),
            "sections": self.sections
        }

class PromptLogger:
    def __init__(self, log_path="prompt_history.jsonl"):
        self.log_path = log_path

    def log(self, prompt_data):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(prompt_data, indent=2) + "\n")

class PromptValidator:
    REQUIRED_SECTIONS = ["Dataset Description", "Task", "Expected Output Format"]
    REQUIRED_PHRASES = [
        "Do not include markdown",
        "Encapsulate logic in a function",
        "if __name__ == '__main__'",
        "raise a Python exception",
        "base64-encoded data URI"
    ]

    def __init__(self, prompt_text):
        self.prompt = prompt_text

    def validate_structure(self):
        missing = [s for s in self.REQUIRED_SECTIONS if f"### {s}" not in self.prompt]
        return {"missing_sections": missing}

    def validate_constraints(self):
        missing = [p for p in self.REQUIRED_PHRASES if p not in self.prompt]
        return {"missing_constraints": missing}

    def validate(self):
        return {
            **self.validate_structure(),
            **self.validate_constraints()
        }

class PromptDebugger:
    def __init__(self, prompt_text, llm_output=None, error_trace=None):
        self.prompt = prompt_text
        self.output = llm_output or ""
        self.error = error_trace or ""

    def detect_common_issues(self):
        issues = []
        if "```" in self.output:
            issues.append("Output contains markdown formatting.")
        if "__main__" not in self.output:
            issues.append("Missing top-level execution block.")
        if any(fallback in self.output for fallback in ["N/A", "Error:", "None"]):
            issues.append("Fallback values returned instead of exceptions.")
        if "data:image" not in self.output and "base64" in self.prompt:
            issues.append("Expected base64-encoded image URI missing.")
        if not self.output.strip().startswith("{"):
            issues.append("Output is not a valid JSON object.")
        return issues

    def suggest_refinements(self):
        issues = self.detect_common_issues()
        suggestions = []
        for issue in issues:
            if "markdown" in issue:
                suggestions.append("Add: 'Do not include markdown formatting.'")
            if "__main__" in issue:
                suggestions.append("Add: 'Always include a top-level execution block using `if __name__ == '__main__'`.'")
            if "Fallback" in issue:
                suggestions.append("Add: 'Do NOT return fallback values like \"N/A\" or \"Error\".'")
            if "base64" in issue:
                suggestions.append("Clarify: 'Return the plot as a base64-encoded data URI.'")
            if "JSON" in issue:
                suggestions.append("Add: 'Only return a valid JSON object with the answers.'")
        return suggestions

    def debug_report(self):
        return {
            "issues_detected": self.detect_common_issues(),
            "suggested_prompt_fixes": self.suggest_refinements()
        }