#!/usr/bin/env python3
"""
scripts/audit_gfm_math.py - Strict GFM / KaTeX Math Compliance Auditor
Enforces all 22 rules from github-math-compliance to guarantee 0 KaTeX rendering errors
and 0 CommonMark parser collisions on GitHub.

Usage:
    python scripts/audit_gfm_math.py [file1.md file2.md ...]
    python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md

Exit Code:
    0 if all files pass with 0 violations.
    1 if one or more violations are detected.
"""

import sys
import os
import re
from typing import List, Tuple, Dict, Any

class MathComplianceAuditor:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.violations: List[Tuple[str, int, str, str, str]] = []

    def audit_file(self, filepath: str) -> List[Tuple[str, int, str, str, str]]:
        file_violations: List[Tuple[str, int, str, str, str]] = []
        if not os.path.exists(filepath):
            file_violations.append((filepath, 0, "FILE_NOT_FOUND", f"File not found: {filepath}", ""))
            self.violations.extend(file_violations)
            return file_violations

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        in_code_block = False
        code_fence = ""
        is_math_block = False
        in_display_math_dollar = False
        prev_line_empty = False

        display_math_buffer: List[str] = []
        display_math_start_line = 0

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Check for display math $$ blocks
            if not in_code_block:
                if stripped.startswith("$$"):
                    if stripped == "$$":
                        if not in_display_math_dollar:
                            # Opening $$ block
                            if line.startswith(" ") or line.startswith("\t"):
                                file_violations.append((filepath, idx, "RULE-08", "Indented display math '$$' block; must start at Column 0", line.rstrip()))
                            in_display_math_dollar = True
                            display_math_start_line = idx
                            display_math_buffer = []
                            prev_line_empty = False
                            continue
                        else:
                            # Closing $$ block
                            if line.startswith(" ") or line.startswith("\t"):
                                file_violations.append((filepath, idx, "RULE-08", "Indented display math '$$' block; must start at Column 0", line.rstrip()))
                            full_display_math = "\n".join(display_math_buffer)
                            self._audit_math_content(filepath, display_math_start_line, full_display_math, is_display=True, violations=file_violations)
                            in_display_math_dollar = False
                            display_math_buffer = []
                            prev_line_empty = False
                            continue
                    else:
                        # Single-line $$ ... $$
                        if line.startswith(" ") or line.startswith("\t"):
                            file_violations.append((filepath, idx, "RULE-08", "Indented display math '$$' block; must start at Column 0", line.rstrip()))
                        math_content = stripped[2:-2] if stripped.endswith("$$") and len(stripped) >= 4 else stripped[2:]
                        self._audit_math_content(filepath, idx, math_content, is_display=True, violations=file_violations)
                        prev_line_empty = False
                        continue

                if in_display_math_dollar:
                    display_math_buffer.append(line.rstrip("\n"))
                    prev_line_empty = False
                    continue

            # Check for fenced code block toggle
            fence_match = re.match(r'^([ \t]*)(```|~~~)(\w*)', line)
            if fence_match:
                indent, fence, tag = fence_match.groups()
                if not in_code_block:
                    in_code_block = True
                    code_fence = fence
                    if tag.lower() in ("math", "katex", "latex-math"):
                        is_math_block = True
                        display_math_start_line = idx
                        display_math_buffer = []
                        if len(indent) > 0:
                            file_violations.append((filepath, idx, "RULE-08", f"Indented '```{tag}' block; display math must start at Column 0", line.rstrip()))
                    else:
                        is_math_block = False
                    prev_line_empty = False
                    continue
                else:
                    if fence == code_fence:
                        if is_math_block:
                            full_display_math = "\n".join(display_math_buffer)
                            self._audit_math_content(filepath, display_math_start_line, full_display_math, is_display=True, violations=file_violations)
                        in_code_block = False
                        code_fence = ""
                        is_math_block = False
                        display_math_buffer = []
                        prev_line_empty = False
                        continue

            if in_code_block:
                if is_math_block:
                    display_math_buffer.append(line.rstrip("\n"))
                prev_line_empty = False
                continue

            # Outside code blocks & display math
            # Rule 12: 4+ leading spaces after blank line
            if prev_line_empty and re.match(r'^[ ]{4,}\S', line):
                file_violations.append((filepath, idx, "RULE-12", "Non-code line with >= 4 spaces leading indent after blank line (triggers unintended code block)", line.rstrip()))

            # Rule 7: \hline
            if r"\hline" in line:
                file_violations.append((filepath, idx, "RULE-07", "Forbidden '\\hline' in markdown table; use native GFM table format", line.rstrip()))

            # Rule 10: List items starting with inline math before bold title
            if re.match(r'^[ \t]*(?:[-*+]|\d+\.)[ \t]+\$', line):
                file_violations.append((filepath, idx, "RULE-10", "List item begins with inline math before a bold title; put bold label first", line.rstrip()))

            # Rule 17: Inline $math$ inside markdown link anchor text [...](...)
            link_match = re.search(r'\[([^\]\n]*\$[^\]\n]*?)\]\([^\)\n]*\)', line)
            if link_match:
                file_violations.append((filepath, idx, "RULE-17", f"Inline math inside markdown link anchor text: '{link_match.group(0)}'; use clean Unicode symbols", line.rstrip()))

            # Rule 14: Parenthesized math ($...$) containing inner parentheses
            paren_math_match = re.search(r'\(\$([^$\n]*\([^$\n]*\)[^$\n]*)\$\)', line)
            if paren_math_match:
                file_violations.append((filepath, idx, "RULE-14", f"Outer markdown parentheses around math with inner parens; move parens inside math '$(...)$'", line.rstrip()))

            # Rule 1: Backtick math
            if re.search(r'(?:\$`|`\$|\$\s*`|`\s*\$)', line):
                file_violations.append((filepath, idx, "RULE-01", "Forbidden backtick math '$`' or '`$'; use clean single '$'", line.rstrip()))

            # Rule 2: Markdown bold or italics wrapping math
            if re.search(r'\*\*[^*$\n]*\$[^$\n]+\$[^*$\n]*\*\*', line):
                file_violations.append((filepath, idx, "RULE-02", "Markdown bold '**...**' wrapping math expression; remove bold formatting around math", line.rstrip()))
            if re.search(r'(?<!\*)\*[^* $\t\n][^*$\n]*\$[^$\n]+\$[^*$\n]*\*(?!\*)', line):
                file_violations.append((filepath, idx, "RULE-02", "Markdown italic '*...*' wrapping math expression", line.rstrip()))
            if re.search(r'__[^_$\n]*\$[^$\n]+\$[^_$\n]*__', line):
                file_violations.append((filepath, idx, "RULE-02", "Markdown bold '__...__' wrapping math expression", line.rstrip()))

            # Rule 19: Flanking hyphen delimiter collision (e.g. low-$\ell$)
            if re.search(r'[a-zA-Z0-9]--?\$', line):
                file_violations.append((filepath, idx, "RULE-19", "Hyphen adjoining inline math delimiter (e.g. 'low-$l$'); use '$\\text{low-}l$' instead", line.rstrip()))

            # Remove escaped dollars and code spans for inline math extraction
            line_no_code = re.sub(r'`[^`\n]*`', '', line)
            dollars = re.findall(r'(?<!\\)\$', line_no_code)
            if len(dollars) % 2 != 0:
                file_violations.append((filepath, idx, "RULE-06", f"Unclosed inline math '$' delimiter on line (count={len(dollars)})", line.rstrip()))

            # Extract inline math spans
            math_spans: List[Tuple[str, int, int]] = []
            for m in re.finditer(r'(?<!\\)\$(.+?)(?<!\\)\$', line_no_code):
                math_spans.append((m.group(1), m.start(), m.end()))

            # Rule 9: >= 2 separate inline $..$ spans with _ separated by prose
            subscript_spans = [s for s in math_spans if '_' in s[0]]
            if len(subscript_spans) >= 2:
                has_intervening_prose = False
                for s_i in range(len(subscript_spans) - 1):
                    end_curr = subscript_spans[s_i][2]
                    start_next = subscript_spans[s_i + 1][1]
                    intervening = line_no_code[end_curr:start_next].strip()
                    if len(intervening) > 0 and not intervening.startswith('$'):
                        has_intervening_prose = True
                        break
                if has_intervening_prose:
                    file_violations.append((filepath, idx, "RULE-09", f"Multiple ({len(subscript_spans)}) inline math spans with subscripts separated by prose on same line; consolidate into single math span or Column-0 display math fence", line.rstrip()))

            # Rule 5: Raw | inside table math cells
            is_table_row = '|' in line
            if is_table_row:
                for span_text, _, _ in math_spans:
                    if re.search(r'(?<!\\)\|', span_text):
                        file_violations.append((filepath, idx, "RULE-05", f"Raw '|' inside table math cell: '${span_text}$'; use '\\lvert', '\\rvert', '\\mid', or '\\Vert'", line.rstrip()))

            # Audit content of each inline math span
            for span_text, _, _ in math_spans:
                self._audit_math_content(filepath, idx, span_text, is_display=False, violations=file_violations)

            prev_line_empty = (len(stripped) == 0)

        if in_display_math_dollar:
            file_violations.append((filepath, display_math_start_line, "RULE-06", "Unclosed display math '$$' block at end of file", ""))
        if in_code_block and is_math_block:
            file_violations.append((filepath, display_math_start_line, "RULE-06", "Unclosed display math '```math' block at end of file", ""))

        self.violations.extend(file_violations)
        return file_violations

    def _audit_math_content(self, filepath: str, line_num: int, math_str: str, is_display: bool, violations: List[Tuple[str, int, str, str, str]]):
        # Rule 3: \text{--}
        if re.search(r'\\text\{[^}]*?--[^}]*?\}', math_str):
            violations.append((filepath, line_num, "RULE-03", "Forbidden '\\text{--}' en-dash ligature in math; use '\\text{-}' or Unicode en-dash '–'", math_str[:80]))

        # Rule 4: Raw * in math mode
        if re.search(r'(?<!\\)\*', math_str):
            violations.append((filepath, line_num, "RULE-04", "Raw '*' inside math mode; use '\\ast', '\\star', '\\times', or '\\cdot'", math_str[:80]))

        # Rule 11: \operatorname
        if re.search(r'\\operatorname(?![a-zA-Z])', math_str):
            violations.append((filepath, line_num, "RULE-11", "Forbidden '\\operatorname' macro (blocked by GitHub KaTeX security filter); use '\\mathrm{...}' or standard operators", math_str[:80]))

        # Rule 13: Lie group subscripts & \widetilde{\mathrm{...}}
        if re.search(r'\\(?:mathrm|mathfrak|mathbf)\{(?:SL|PSL|GL|SU|SO)\}_|\\widetilde\{\\mathrm\{|\\mathrm\{SO\}_|\\mathrm\{SU\}_|\\mathrm\{SL\}_', math_str):
            violations.append((filepath, line_num, "RULE-13", "Lie group underscore subscript or '\\widetilde{\\mathrm{...}}'; use '\\tilde{\\mathrm{SL}}(2, \\mathbb{R})', '\\mathrm{SO}(3)', etc.", math_str[:80]))

        # Rule 15: Unescaped \left\{ or \right\}
        if re.search(r'\\(?:left|right)\\[\{\}]', math_str):
            violations.append((filepath, line_num, "RULE-15", "Unescaped '\\left\\{' or '\\right\\}'; use '\\left\\lbrace' and '\\right\\rbrace'", math_str[:80]))

        # Rule 16: Brace-preceded font macro subscripts
        if re.search(r'\\(?:mathcal|mathbb|mathbf|mathfrak|vec)\{[A-Za-z0-9]\}_', math_str):
            violations.append((filepath, line_num, "RULE-16", "Brace-preceded font macro subscript (e.g. '\\mathcal{H}_F'); drop braces before subscript ('\\mathcal H_F') to avoid CommonMark underscore emphasis hijack", math_str[:80]))

        # Rule 18: Prime-preceded subscripts
        if re.search(r'\\(?:bigotimes|prod|coprod|bigoplus|sum)\'\_', math_str):
            violations.append((filepath, line_num, "RULE-18", "Prime-preceded subscript (e.g. '\\bigotimes'_p'); use '{\\bigotimes_p}'' or Column-0 display math fence", math_str[:80]))

        # Rule 20: Unescaped literal & in \text{} or \mathrm{}
        if re.search(r'\\(?:text|mathrm)\{[^}]*?(?<!\\)&[^}]*?\}', math_str):
            violations.append((filepath, line_num, "RULE-20", "Unescaped literal '&' inside '\\text{...}'; escape as '\\&' or write 'and'", math_str[:80]))

        # Rule 21: _ or \textunderscore in \text{} or \mathrm{}
        if re.search(r'\\(?:text|mathrm)\{[^}]*?(?:(?<!\\)_|\\textunderscore)[^}]*?\}', math_str):
            violations.append((filepath, line_num, "RULE-21", "Forbidden '_' or '\\textunderscore' inside '\\text{...}'; isolate code identifiers in backticks outside math or use hyphens", math_str[:80]))

        # Rule 22: Unsupported \llbracket or \rrbracket
        if re.search(r'\\(?:llbracket|rrbracket)', math_str):
            violations.append((filepath, line_num, "RULE-22", "Unsupported '\\llbracket' / '\\rrbracket' macro; use '\\lbrack\\!\\lbrack ... \\rbrack\\!\\rbrack'", math_str[:80]))

def main():
    target_files = sys.argv[1:]
    if not target_files:
        candidate_files = ["README.md", "docs/ARCHITECTURE.md"]
        target_files = [f for f in candidate_files if os.path.exists(f)]
        if not target_files:
            print("Usage: python scripts/audit_gfm_math.py <file1.md> [file2.md ...]")
            sys.exit(1)

    auditor = MathComplianceAuditor(verbose=True)
    total_violations = 0

    print("=" * 80)
    print(" AstroSnail GFM & KaTeX Math Compliance Audit (22 Strict Rules)")
    print("=" * 80)

    for filepath in target_files:
        print(f"Auditing: {filepath} ...")
        file_violations = auditor.audit_file(filepath)
        total_violations += len(file_violations)
        if file_violations:
            for v_file, v_line, v_rule, v_msg, v_snippet in file_violations:
                print(f"  [ERROR] {v_file}:{v_line} [{v_rule}]: {v_msg}")
                if v_snippet:
                    print(f"          Snippet: {v_snippet[:100]}")
        else:
            print(f"  [PASS] 0 violations detected.")

    print("-" * 80)
    print(f"Audit Summary: {len(target_files)} files scanned, {total_violations} total violations.")
    print("=" * 80)

    if total_violations > 0:
        print("FAILED: Math compliance audit failed. Please fix all reported issues.")
        sys.exit(1)
    else:
        print("SUCCESS: All mathematical expressions are 100% GFM/KaTeX compliant!")
        sys.exit(0)

if __name__ == "__main__":
    main()
