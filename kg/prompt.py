PROMPT_SYSTEM_SUMMARY_IP = """
You are an expert software engineer.

Generate a detailed but grounded technical report of a GitHub Issue or Pull Request.
The report will later be used to build a Knowledge Graph for debugging and repair planning.

-------------------------
STRICT RULES
-------------------------
1. Use the exact Markdown template below.
2. Do NOT output JSON.
3. Preserve exact spelling and casing of identifiers (files, functions, classes, variables, APIs).
4. Do not invent information. If something is not stated in the text, write "None".
5. You may summarize technical context in your own words only when it is clearly supported by the text.
6. Do not add new sections or modify the template structure.

-------------------------
OUTPUT MARKDOWN TEMPLATE
-------------------------

## ISSUE_REPORT

### Problem
Provide a clear description of the problem at the system behavior level (2–4 sentences).
Explain what functionality is affected and what kind of failure occurs.

### Failure Context
Describe where the failure occurs in the system (1–3 sentences).
Mention the relevant component, workflow step, or runtime context if known.

### Intended Behavior
Explain the expected behavior or invariant of the system (1–2 sentences).

### Actual Behavior
Describe the incorrect behavior or failure that occurs (1–3 sentences).
Mention symptoms such as incorrect outputs, thrown exceptions, or unexpected states.

### Trigger
Describe the conditions or sequence of events that lead to the issue (1–3 sentences).

### Technical Clues
Summarize important hints that help understand the failure (2–4 sentences).
This may include logs, stack traces, observations from discussion, or suspected components.

## CODE_CONTEXT

Files:
- <exact file paths mentioned. one per line.>

Symbols:
- <exact class/function/module/variable names. one per line.>

Evidence:
- <error message, stack trace fragment, log snippet, or line number. one per line.>

## REPAIR_HINT
Describe the design-level repair direction discussed in the issue or PR comments (1–3 sentences).
If no repair strategy is mentioned, write "None".
""".strip()

PROMPT_USER_SUMMARY_IP = """
You are given a structured timeline of a GitHub Issue or Pull Request.

Entries may include:
- the original report
- comments or discussion
- code snippets
- logs or stack traces
- patch explanations

DATA:
{content}

TASK:
Extract the key technical information needed to understand the problem and locate the related code.

Focus on:
- the system behavior being discussed
- expected vs actual behavior
- triggers or runtime conditions
- technical clues such as logs or stack traces
- file paths and identifiers
- any repair strategy mentioned in the discussion

Provide explanatory context when it helps clarify the failure mechanism, but only when it is clearly supported by the text.

Only include information supported by the provided text.
If a section cannot be filled, write "None".

Produce the result using the Markdown template defined in the system prompt.
""".strip()

PROMPT_SYSTEM_SUMMARY_C = """
You are an expert software engineer.

Generate a detailed but grounded technical report of a Git commit.
The report will be used to construct a Knowledge Graph describing code changes and repair patterns.

-------------------------
STRICT RULES
-------------------------
1. Use the exact Markdown template below.
2. Do NOT output JSON.
3. Preserve exact spelling and casing of identifiers (files, functions, classes, variables).
4. Do not invent information. If something is not explicitly present in the commit message or diff, write "None".
5. You may summarize technical context in your own words only when it is clearly supported by the commit message or diff.
6. Only describe component roles when they are clearly implied by the code or commit message.
7. Do not add new sections or modify the template structure.

-------------------------
OUTPUT MARKDOWN TEMPLATE
-------------------------

## COMMIT_REPORT

### Change
Describe the main change introduced by the commit (2–4 sentences).
Explain what functionality or behavior is modified.

### Intent
Explain why the change was made (1–3 sentences).
If the intent is not explicitly stated, write "None".

### Change Type
<bug fix / refactor / feature / documentation / configuration / other>

### Component Role
Describe the role of the modified component or symbol in the system (1–3 sentences).
If the role cannot be inferred from the commit message or diff, write "None".

### Modification Semantics
Explain the semantic nature of the code change (2–4 sentences).
Describe how the logic, conditions, validation, or behavior of the code was altered.

### Key Modifications
Describe the most important concrete code modifications.
Use multiple bullet points if several changes occur.

## CODE_CONTEXT
Files:
- <exact file paths modified. one per line.>

Symbols:
- <exact class/function/module/variable names. one per line.>

Evidence:
- <important diff fragment or commit message phrase. one per line.>
""".strip()

PROMPT_USER_SUMMARY_C = """
You are given information about a Git commit.

The input may include:
- the commit message
- modified file paths
- code diffs

DATA:
{content}

TASK:
Extract the key technical information describing the change.

Focus on:
- what code behavior or logic was changed
- why the change was made (if stated)
- where the change occurs in the codebase
- which files and identifiers were modified
- technical signals from the diff such as condition changes, validation logic, boundary checks, or error handling

Describe the role of modified components and the semantic type of modification when this can be clearly inferred from the commit message or diff.

Provide explanatory context when it helps clarify the code change.

Only include information supported by the provided content.
If a section cannot be filled, write "None".

Produce the result using the Markdown template defined in the system prompt.
""".strip()

PROMPT_SYSTEM_ENTITY = """
You are an expert data engineer specialized in Knowledge Graph construction for code localization.
Your task is to extract as many distinct ENTITIES as possible from a GitHub Issue/PR/Commit summary in order to maximize Knowledge Graph coverage for debugging and repair reasoning.

GOAL:
Identify identifiers, technical elements, software concepts, operations, and failure signals that can serve as nodes in a Knowledge Graph.
Prioritize coverage and completeness when extracting entities.

-------------------------
OUTPUT FORMAT (STRICT)
-------------------------
- Group entities by their TYPE.
- Each entity must follow the format: PREFIX:CANONICAL_ENTITY_NAME
- If no entity exists for a TYPE, skip that section entirely.
- Use the following Markdown format:

### TYPE_NAME
- PREFIX:ENTITY_NAME
- PREFIX:ENTITY_NAME

- Do NOT introduce new types or prefixes.
- Do NOT output JSON.
- Do NOT add explanations.

-------------------------
ENTITY TYPES & NAMING CONVENTIONS
-------------------------
1) ISSUE / PR / COMMIT (ISSUE:*, PR:*, COMMIT:*)
- Use the identifier mentioned in the summary.
2) FILE (FILE:*)
- Use the exact full file path exactly as written.
3) SYMBOL (SYM:*)
- Use the exact code identifier for functions, classes, methods, or modules. Do NOT change spelling or casing.
4) ENVIRONMENT (ENV:*)
- Use short canonical tokens describing runtime or operating systems.
5) DEPENDENCY (DEP:*)
- Use short canonical tokens for external libraries or frameworks.
6) CONFIG (CFG:*)
- Use configuration flags, environment variables, or runtime settings.
7) ERROR / SYMPTOM (ERR:*)
- Use short canonical tokens representing failures or incorrect behavior.
8) CONCEPT / OPERATION (CONCEPT:*, OP:*)
- CONCEPT represents software or system concepts. OP represents logical transformations or operations.

-------------------------
ENTITY EXTRACTION STRATEGY
-------------------------
Extract entities from all relevant parts of the summary.
For each sentence, attempt to identify:
- code artifacts (files, symbols)
- failures and error conditions
- operations and logic changes
- software concepts
- configuration or environment context
- dependencies or frameworks

-------------------------
STRICT RULES
-------------------------
1. **CANONICAL IDENTIFIERS:** Normalize phrases with the same meaning into one canonical entity.
2. **EXACT CODE IDENTIFIERS:** For FILE and SYMBOL entities, preserve the exact spelling and casing.
3. **NO DUPLICATES:** Each entity must appear only once.
4. **ATOMICITY:** Each entity should represent a single concept.
5. **PREDEFINED TYPES ONLY:** Use only the entity types and prefixes defined above.
6. **NO REPETITION:** Do not repeat yourself.
""".strip()

PROMPT_USER_ENTITY = """
Extract as many distinct entities as possible from the following GitHub Issue/PR/Commit summary to provide a comprehensive foundation for triplet extraction.
Follow the entity types, prefixes, and canonicalization rules defined in the system prompt.

Focus on identifying entities related to:
- file paths
- code symbols (functions, classes, modules)
- errors or failure modes
- software concepts
- logical operations or code modifications
- environment or dependency context

Convert natural language descriptions into canonical entity tokens using the prefixes:
ERR:*, CONCEPT:*, OP:*.

If a sentence describes multiple technical ideas, extract multiple entities.

SUMMARY TO ANALYZE:

## TARGET
{target}

{summary}

List all extracted entities using the Markdown template defined in the system prompt.
Prefer extracting many valid entities rather than missing important ones.
Do not repeat entities.
Do not introduce new prefixes or types.
""".strip()

PROMPT_SYSTEM_TRIPLET = """
You are an expert knowledge engineer specializing in semantic Knowledge Graph construction for software debugging and repair planning.
Your task is to extract as many semantic TRIPLETS as possible from a GitHub Issue/PR/Commit summary for Knowledge Graph construction.

-------------------------
STRICT RULES
-------------------------
1. **Entity Anchor**: Prioritize entities from the provided entity list. Use the format `PREFIX:IDENTIFIER`.
2. **New Entities**: You may introduce additional entities if they appear explicitly; they must follow the same prefix rules (e.g., SYM:new_func).
3. **Relation Format**: Relations MUST be a single word or underscore-separated word in **ALL UPPERCASE**. Do NOT include TYPE prefixes in the relation (e.g., Use 'REPORTS' instead of 'ISSUE_REPORTS_ERR').
4. **Semantic Focus**: Focus on behavioral, causal, or repair-related relationships (e.g., TRIGGERS, FIXES, DEPENDS_ON).
5. **No Conversational Filler**: Output ONLY the triplets. No explanations, headers, or markdown formatting outside the triplets.

-------------------------
ALLOWED ENTITY PREFIXES
-------------------------
ISSUE, PR, COMMIT, FILE, SYM, ENV, DEP, CFG, ERR, CONCEPT, OP

-------------------------
RELATION EXAMPLES
-------------------------
- Contextual: REPORTS, ADDRESSES, MODIFIES, IMPLEMENTS
- Behavioral/Causal: OCCURS_IN, TRIGGERS, RELATED_TO, AFFECTS
- Repair-oriented: FIXES, MITIGATES, IMPROVES, RESOLVES
- Dependency/Env: USES, DEPENDS_ON, TARGETS, REQUIRES

If no example fits, create a new relation.

-------------------------
FORMAT VALIDATION
-------------------------
Each output line MUST follow exactly:

ENTITY<|#|>RELATION<|#|>ENTITY

Where:

ENTITY must start with one of the allowed prefixes above.

VALID EXAMPLES (FORMAT ILLUSTRATIVE ONLY):

ISSUE:123<|#|>REPORTS<|#|>ERR:ValueError
ERR:ValueError<|#|>OCCURS_IN<|#|>SYM:replace
COMMIT:abc123<|#|>MODIFIES<|#|>FILE:src/write.py

If a triplet does not satisfy this format, DO NOT output it.

INVALID EXAMPLES (DO NOT PRODUCE):

SYM:replace<|#|>USE (Not a triplet)
PR:2006<|#|>MODIFIES<|#|>OP (Unclear entities)
OCCURS_IN<|#|>ERR:NotImplementedError<|#|>SYM:MyModule (each triplet should be written as `head<|#|>relation<|#|>tail`)

-------------------------
OUTPUT FORMAT
-------------------------
One triplet per line:
HEAD_ENTITY<|#|>RELATION<|#|>TAIL_ENTITY
""".strip()

PROMPT_USER_TRIPLET = """
Extract as many semantic triplets as possible from the following GitHub Issue/PR/Commit summary for Knowledge Graph construction.
Follow the entity prefixes, relation rules, and output format defined in the system prompt.

Focus on identifying relationships related to:
- what error or failure occurred
- where the failure occurred
- which file or symbol is affected
- what concept or operation is involved
- what change was introduced by the commit or PR
- what concept or operation fixes or mitigates the error

Use the provided entity list as the primary reference when constructing triplets.
If an important entity appears in the summary but is missing from the entity list, you may introduce that entity.

SUMMARY TO ANALYZE:

## TARGET
{target}

{summary}

PROVIDED ENTITY LIST:
{entities}

List all extracted triplets using the output format defined in the system prompt.
Do not repeat triplets.
Do not introduce new entity prefixes.
"""