"""Agent-shaped conversations.

What determines cache behaviour is the *structure* of the message list, not
whether the tools really exist. So the harness builds exactly what a coding
agent sends: a system prompt with tool definitions, a user instruction, and
then alternating assistant messages carrying tool calls and tool messages
carrying file contents or test output. Fake tools with plausible output are
enough.

The prefix is deliberately layered, cheapest-to-invalidate last:

    [system prompt + tool defs]   identical for every student, always
    [shared assignment brief]     identical for every student, always
    [shared project skeleton]     identical for every student, shared_fraction
    [student's own files]         per student
    [live agent steps]            grows during the run

Everything above the first per-student byte is a cache hit across the whole
class. That is why ``shared_fraction`` is its own axis in the matrix.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Sequence

from .corpus import CodeCorpus, SourceFile
from .personas import DEFAULT_WORK_PROFILES, WorkProfile
from .tokens import MESSAGE_OVERHEAD_TOKENS, TokenCounter

SYSTEM_PROMPT = """You are a coding assistant working inside a student's development environment.
You help a first- or second-year software development student build and debug a small web application.

Working agreements:
- Read files before you change them. Never guess at file contents.
- Make the smallest change that solves the problem, then run the tests.
- When a test fails, read the failure output before editing again.
- Explain what you changed in one or two sentences, in plain language.
- Prefer the conventions already present in the project over your own preferences.
- Do not invent files, functions or packages that you have not seen.
- If the request is ambiguous, make the most reasonable assumption and say which one you made.

You have access to the tools described below. Call them one at a time and wait
for the result before deciding the next step. Stop calling tools once the task
is done and write a short summary for the student.

Environment:
- Editor: VS Code with the workspace open at the project root
- Shell: bash
- The test suite is run through the project's own runner
- The student is working on the assignment described in the first user message
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace, optionally a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root"},
                    "start_line": {"type": "integer", "description": "First line to read, 1-based"},
                    "end_line": {"type": "integer", "description": "Last line to read, inclusive"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact block of text in a file with new text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Text to replace, must match exactly"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the project test suite, optionally filtered to one test path.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "Optional test file or name filter"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the workspace for a literal string or regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string", "description": "Optional file glob to limit the search"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

# One class, one assignment. The brief is shared because that is the point:
# students in a lesson do not each work on something different.
ASSIGNMENT_BRIEF = """Opdracht 4 - Gebruikersbeheer

Bouw in dit project een werkend gebruikersdeel:

1. Een registratieformulier met validatie op e-mailadres en wachtwoordlengte.
2. Een inlogformulier dat een sessie start en foutmeldingen netjes toont.
3. Een profielpagina waarop een ingelogde gebruiker de eigen gegevens kan wijzigen.
4. Tests voor de validatie en voor het in- en uitloggen.

Voorwaarden: houd je aan de bestaande structuur van het project, schrijf geen
wachtwoorden in platte tekst weg, en zorg dat de bestaande tests blijven slagen.
Lever in via de repository, met een korte toelichting in de README.
"""

STUDENT_INSTRUCTIONS = [
    "Bouw het inlogformulier af, de validatie doet nog niets.",
    "Mijn tests falen met een KeyError, kun je kijken waar dat vandaan komt?",
    "Voeg een profielpagina toe waar je je e-mailadres kunt wijzigen.",
    "De registratie slaat het wachtwoord in platte tekst op, dat moet gehasht.",
    "Ik krijg een 500 als ik uitlog. Wat gaat er mis?",
    "Schrijf tests voor de wachtwoordvalidatie.",
    "De foutmeldingen staan in het Engels, maak er Nederlands van.",
    "Refactor de validatie naar een aparte functie, het staat nu drie keer.",
    "Voeg een 'wachtwoord vergeten'-knop toe die voorlopig alleen een melding toont.",
    "Waarom blijft de sessie niet bewaard na het inloggen?",
    "De layout van het formulier klopt niet op mobiel, kun je dat fixen?",
    "Kun je uitleggen wat deze middleware doet en er commentaar bij zetten?",
]

TEST_OUTPUT_TEMPLATES = [
    """$ pytest -q
{dots}
{failures} failed, {passed} passed in {seconds:.2f}s

=================================== FAILURES ===================================
_____________________________ test_{case} ______________________________

    def test_{case}():
        payload = {{"email": "student@firda.nl", "password": "geheim"}}
        response = client.post("/api/users/login", json=payload)
>       assert response.status_code == 200
E       assert 400 == 200
E        +  where 400 = <Response [400]>.status_code

tests/test_{module}.py:{line}: AssertionError
----------------------------- Captured log call ------------------------------
WARNING  app.auth:auth.py:{line2} validation failed for field 'password'
""",
    """$ npm test

  {module}
    {check} renders the form
    {check} shows a validation message for a short password
    1) submits the form

  {passed} passing ({seconds:.0f}ms)
  {failures} failing

  1) {module} submits the form:
     AssertionError: expected 'undefined' to equal 'student@firda.nl'
      at Context.<anonymous> (test/{module}.test.js:{line}:12)
""",
    """$ php artisan test

   PASS  Tests\\Unit\\ValidationTest
  {check} email must be valid
  {check} password minimum length

   FAIL  Tests\\Feature\\AuthTest
  {check} user can register
  x user can log in
   Tests:  {failures} failed, {passed} passed
   Time:   {seconds:.2f}s

  Failed asserting that 302 is identical to 200.
  at tests/Feature/AuthTest.php:{line}
""",
]

EDIT_ACK = "Applied edit to {path}: 1 replacement, {added} lines added, {removed} lines removed."


@dataclass
class Session:
    """One student's ongoing agent conversation."""

    student_index: int
    corpus: CodeCorpus
    counter: TokenCounter
    target_tokens: int
    shared_fraction: float
    rng: random.Random
    message_style: str = "tool_calls"        # or "text" for servers without tool support
    # How heavy this student's agent steps are: how many files come back per
    # read, how big the search and test output is, how much the model writes.
    # Defaults to the lightest profile so callers that predate work profiles
    # keep the behaviour they had.
    work: WorkProfile = field(default_factory=lambda: DEFAULT_WORK_PROFILES[0])
    messages: list[dict] = field(default_factory=list)
    prompt_tokens_estimate: int = 0
    step_number: int = 0
    burst_number: int = 0
    compactions: int = 0
    restarts: int = 0
    initial_tokens: int = 0
    shared_prefix_tokens: int = 0
    shared_prefix_messages: int = 2
    context_shortfall: int = 0

    # ---------------------------------------------------------------- build

    def reset(self) -> None:
        """Start a fresh, cold session. Used at run start and on a restart."""
        self.messages = []
        self.step_number = 0
        self.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        self.messages.append({"role": "user", "content": ASSIGNMENT_BRIEF})

        base = self.counter.count_messages(self.messages) + self.counter.count_tools(TOOLS)

        # Fill to 70 % of the target; the remaining headroom is what the live
        # agent steps grow into before the session needs compacting.
        initial_budget = int(self.target_tokens * 0.70)
        shared_budget = int(initial_budget * self.shared_fraction)

        group = self.corpus.group(self.work.group)
        used = self._fill_from(group.shared_files, base, shared_budget)
        self.shared_prefix_tokens = used
        # Everything up to here is byte-identical across the class. Compaction
        # must never touch it, or the shared prefix stops being a prefix.
        self.shared_prefix_messages = len(self.messages)
        own_files = group.student_files(self.student_index, 400)
        used = self._fill_from(own_files, used, initial_budget)

        self.prompt_tokens_estimate = used
        self.initial_tokens = used
        self.context_shortfall = max(0, initial_budget - used)

    def _fill_from(self, sources: Sequence[SourceFile], used: int, budget: int) -> int:
        """Append read_file turns until ``budget`` tokens are reached.

        Two rules keep the result honest. A file that would push the context
        past the run's target is skipped rather than appended -- the context
        size is an axis of the matrix, so a "32k run" that quietly became a
        41k run because the last file happened to be large would compare
        against the wrong column. And if the pool runs out before the budget
        does -- a 100k context with a 90 % shared skeleton needs more code
        than a small project contains -- we keep going with line-range reads
        of the same files. A real agent re-reads regions all the time, and
        the bytes stay deterministic, which is what the shared prefix
        depends on.
        """
        if not sources or used >= budget:
            return used
        ceiling = self.target_tokens
        index = 0
        skipped_in_a_row = 0
        guard = 0
        while used < budget and guard < 5000:
            guard += 1
            if skipped_in_a_row >= len(sources):
                break            # nothing left that still fits
            source = sources[index % len(sources)]
            pass_number = index // len(sources)
            index += 1
            if pass_number == 0:
                text = source.content
                start_line = end_line = None
            else:
                lines = source.content.splitlines()
                if len(lines) < 6:
                    skipped_in_a_row += 1
                    continue
                window = max(20, len(lines) // 3)
                start = (pass_number * window) % max(1, len(lines) - 3)
                end = min(len(lines), start + window)
                if end - start < 3:
                    skipped_in_a_row += 1
                    continue
                text = "\n".join(lines[start:end])
                start_line, end_line = start + 1, end
            cost = self.counter.count(text) + 2 * MESSAGE_OVERHEAD_TOKENS + 24
            if used + cost > ceiling:
                skipped_in_a_row += 1
                continue
            skipped_in_a_row = 0
            used += self._append_file_read(source, start_line, end_line,
                                           None if start_line is None else text)
        return used

    def _append_file_read(self, source: SourceFile, start_line: int | None = None,
                          end_line: int | None = None, text: str | None = None) -> int:
        before = len(self.messages)
        # The id must not contain the student index: the shared skeleton block
        # has to be byte-identical across the class or it is not a shared prefix.
        call_id = f"call_{before:04d}"
        payload: dict = {"path": source.path}
        if start_line is not None:
            payload["start_line"] = start_line
            payload["end_line"] = end_line
        arguments = json.dumps(payload, separators=(",", ":"))
        body = text if text is not None else source.content
        header = source.path if start_line is None else f"{source.path}:{start_line}-{end_line}"
        result = f"# {header}\n{body}"
        if self.message_style == "tool_calls":
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "read_file", "arguments": arguments},
                }],
            })
            self.messages.append({
                "role": "tool", "tool_call_id": call_id, "name": "read_file",
                "content": result,
            })
        else:
            self.messages.append({
                "role": "assistant",
                "content": f"<tool_call>read_file({arguments})</tool_call>",
            })
            self.messages.append({
                "role": "user",
                "content": f"<tool_result name=\"read_file\">\n{result}\n</tool_result>",
            })
        return self.counter.count_messages(self.messages[before:])

    # ----------------------------------------------------------------- turn

    def start_burst(self, intensity: float = 1.0) -> None:
        """A new student instruction. This is where compaction happens."""
        self.burst_number += 1
        if self.prompt_tokens_estimate > self.target_tokens:
            self._compact()
        instruction = self.rng.choice(STUDENT_INSTRUCTIONS)
        self.messages.append({"role": "user", "content": instruction})
        self.prompt_tokens_estimate += self.counter.count(instruction) + 4

    def _compact(self) -> None:
        """What a real agent does when the window fills up: drop the oldest
        part of the working history and keep the head and the recent turns.

        Two rules make this honest rather than convenient. The shared prefix
        is preserved, because that is what a real agent's system prompt and
        project context do. And the cut never lands on a tool result, which
        would leave a tool message without the assistant turn that called it
        and make the conversation invalid.

        Everything after the preserved head loses its cache entry -- exactly
        the cost we want to measure -- so this happens on burst boundaries
        only, never in the middle of one.
        """
        head = max(self.shared_prefix_messages, 2)
        keep_tail = 8
        if len(self.messages) <= head + keep_tail + 4:
            return
        body = self.messages[head:len(self.messages) - keep_tail]
        cut = len(body) // 2
        while cut < len(body) and body[cut].get("role") == "tool":
            cut += 1
        if cut >= len(body):
            return
        self.messages = self.messages[:head] + body[cut:] + self.messages[-keep_tail:]
        self.prompt_tokens_estimate = self.counter.count_messages(self.messages)
        self.compactions += 1

    def next_step(self) -> tuple[list[dict], int]:
        """Return the message list to send plus a max_tokens for this step."""
        self.step_number += 1
        max_tokens = self._step_output_tokens()
        return list(self.messages), max_tokens

    def _step_output_tokens(self) -> int:
        """Tool-calling steps are short, the closing summary is long.

        Both ranges come from the work profile. This is the number that decides
        how much decode work one instruction costs, so it is the last place
        that should carry a hard-coded guess.
        """
        if self.rng.random() < 0.25:
            low, high = self.work.output_tokens_final   # explanation / final answer
        else:
            low, high = self.work.output_tokens_tool    # a tool call with arguments
        return self.rng.randint(int(low), max(int(low), int(high)))

    def record_step(self, assistant_text: str) -> None:
        """Append the model's answer plus the tool result it triggered."""
        tool_name, arguments, result = self._synthesise_tool_result()
        call_id = f"call_{len(self.messages):04d}"
        if self.message_style == "tool_calls":
            self.messages.append({
                "role": "assistant",
                "content": assistant_text[:400] or None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }],
            })
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "name": tool_name, "content": result})
        else:
            self.messages.append({
                "role": "assistant",
                "content": (assistant_text[:400] + "\n" if assistant_text else "")
                           + f"<tool_call>{tool_name}({arguments})</tool_call>",
            })
            self.messages.append({
                "role": "user",
                "content": f"<tool_result name=\"{tool_name}\">\n{result}\n</tool_result>",
            })
        self.prompt_tokens_estimate += self.counter.count_messages(self.messages[-2:])

    def _synthesise_tool_result(self) -> tuple[str, str, str]:
        """One tool call and its result, sized by this student's work profile.

        The four tools are the ones a coding agent actually has. What differs
        per profile is the weight of each call: a student tweaking a form gets
        one small file back, a student having the agent grind through a Unity
        project gets several large ones, a grep with context lines, and a test
        run with a handful of failures.
        """
        group = self.corpus.group(self.work.group)
        mix = self.work.tool_mix or {"read_file": 1.0}
        names = list(mix)
        tool = self.rng.choices(names, weights=[max(mix[n], 0.0) for n in names])[0]

        if tool == "read_file":
            return self._read_files(group)
        if tool == "edit_file":
            source = self._pick(group, 1)[0]
            snippet = source.content[:600]
            arguments = json.dumps(
                {"path": source.path, "old_text": snippet[:120], "new_text": snippet[:120]},
                separators=(",", ":"))
            return "edit_file", arguments, EDIT_ACK.format(
                path=source.path, added=self.rng.randint(1, 30),
                removed=self.rng.randint(0, 12))
        if tool == "run_tests":
            return "run_tests", json.dumps({"target": ""}), self._test_output()
        return self._search(group)

    def _pick(self, group, count: int) -> list[SourceFile]:
        """``count`` files from this student's pool, chosen per call."""
        pool = group.student_files(self.student_index, 40)
        return [pool[self.rng.randrange(0, len(pool))] for _ in range(max(1, count))]

    def _read_files(self, group) -> tuple[str, str, str]:
        low, high = self.work.files_per_read
        count = self.rng.randint(int(low), max(int(low), int(high)))
        sources = self._pick(group, count)
        budget_low, budget_high = self.work.read_line_budget
        parts, paths = [], []
        for source in sources:
            paths.append(source.path)
            if budget_high and budget_high > 0:
                # A real agent often reads a window rather than a whole file.
                lines = source.content.splitlines()
                span = self.rng.randint(int(budget_low) or 1, int(budget_high))
                start = self.rng.randrange(0, max(1, len(lines)))
                body = "\n".join(lines[start:start + span])
                parts.append(f"# {source.path}:{start + 1}\n{body}")
            else:
                parts.append(f"# {source.path}\n{source.content}")
        arguments = json.dumps({"path": paths[0]} if len(paths) == 1
                               else {"paths": paths}, separators=(",", ":"))
        return "read_file", arguments, "\n\n".join(parts)

    def _search(self, group) -> tuple[str, str, str]:
        pattern = self.rng.choice(["password", "session", "validate", "login", "email",
                                   "Update", "Awake", "Serialize", "collider"])
        low, high = self.work.search_matches
        wanted = self.rng.randint(int(low), max(int(low), int(high)))
        pool = group.student_files(self.student_index, max(wanted, 12))
        context_lines = max(0, int(self.work.search_context_lines))
        blocks = []
        for i in range(wanted):
            source = pool[i % len(pool)]
            lines = source.content.splitlines() or [""]
            hit = self.rng.randrange(0, len(lines))
            if context_lines:
                lo = max(0, hit - context_lines)
                hi = min(len(lines), hit + context_lines + 1)
                body = "\n".join(f"{lo + n + 1}:{text}"
                                  for n, text in enumerate(lines[lo:hi]))
                blocks.append(f"{source.path}\n{body}")
            else:
                blocks.append(f"{source.path}:{hit + 1}:{lines[hit][:120]}")
        joiner = "\n--\n" if context_lines else "\n"
        return ("search_code", json.dumps({"pattern": pattern}),
                f"{wanted} matches:\n" + joiner.join(blocks))

    def _test_output(self) -> str:
        """A test run. A bigger project fails in more places at once, so the
        profile decides how many failure blocks come back -- the difference
        between a 70-token result and a 700-token one."""
        low, high = self.work.test_output_blocks
        blocks = self.rng.randint(int(low), max(int(low), int(high)))
        return "\n".join(self._one_test_block() for _ in range(blocks))

    def _one_test_block(self) -> str:
        template = self.rng.choice(TEST_OUTPUT_TEMPLATES)
        return template.format(
            dots="." * self.rng.randint(20, 60) + "F",
            failures=self.rng.randint(1, 3),
            passed=self.rng.randint(8, 45),
            seconds=self.rng.uniform(0.4, 9.0),
            case=self.rng.choice(["login_returns_token", "password_too_short", "logout_clears_session"]),
            module=self.rng.choice(["auth", "users", "profile"]),
            line=self.rng.randint(10, 140),
            line2=self.rng.randint(10, 140),
            check="✓",
        )
