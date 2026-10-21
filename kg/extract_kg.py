from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from github import Github, Auth
from datetime import datetime, timezone

from api_github import extract_triplets, extract_commit, format_issue, format_commit
from prompt import *

import numpy as np
import argparse
import asyncio
import pickle
import json
import os
import re

### Define Arguments ###
parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, default="SWE-bench-Live/SWE-bench-Live", help="Target dataset.")
parser.add_argument("-s", type=str, default="lite", help="Target split.")
parser.add_argument("-n", type=int, default=10, help="Number of seed issues.")
parser.add_argument("-hop", type=int, default=1, help="Number of hops.")
parser.add_argument("-m", type=str, default="gpt-4o-mini", help="LLM type.")
parser.add_argument("-w", type=int, default=64, help="Number of workers.")
args = parser.parse_args()
### Arguments ###

def normalize_linesplit(text):
    return "\n".join([line.strip() for line in text.splitlines()])

def extract_number(text):
    if text.startswith("ISSUE:"):
        return int(text[6:])
    elif text.startswith("PR:"):
        return int(text[3:])
    else:
        return None

def is_test_file(file):
    word_list = re.split(r" |_|\/", file.lower())
    return any([word.startswith("test") for word in word_list])

def remove_duplicate(x):
    return list(dict.fromkeys(x))

async def async_extract_triplets(repo, issue, dt, logger):
    async with sem:
        return issue, await asyncio.to_thread(extract_triplets, repo, issue, dt, logger)

async def async_extract_commit(repo, sha, logger):
    async with sem:
        return sha, await asyncio.to_thread(extract_commit, repo, sha, logger)

async def expand_kg(repo, dt, state, hop, triplets, dict_issue, logger):
    seen, unseen = state

    tasks = [async_extract_triplets(repo, issue, dt, logger) for issue in unseen]
    results = await tqdm.gather(*tasks)

    issues, new = seen.union(unseen), set()

    for issue, (tri_issue, dict_content) in results:
        if dict_content:
            dict_issue[str(issue)] = dict_content
        for h, r, t in tri_issue:
            if hop == 0:
                if extract_number(t) and (extract_number(t) not in issues):
                    continue
                triplets.add((h, r, t))
            else:
                triplets.add((h, r, t))
                if extract_number(t) and (extract_number(t) not in issues):
                    new.add(extract_number(t))
    
    if hop == 0:
        return issues, triplets, dict_issue

    return await expand_kg(repo, dt, [issues, new], hop - 1, triplets, dict_issue, logger)

async def gather_commits(repo, list_sha, logger):
    tasks = [async_extract_commit(repo, sha, logger) for sha in list_sha]
    results = await tqdm.gather(*tasks)

    dict_commit = dict()
    for sha, dict_content in results:
        dict_commit[sha[:6]] = dict_content

    return dict_commit

def extract_commit_triplets(dict_commit):
    triplet_commit = []
    if not dict_commit:
        return triplet_commit
    
    for sha in dict_commit:
        if not dict_commit[sha]:
            continue
        if "changes" not in dict_commit[sha]:
            continue

        for change in dict_commit[sha]["changes"]:
            if is_test_file(change["filename"]) or not change["filename"].endswith(".py"):
                continue
            if not change["patch"]:
                continue

            filename, status, patch = change["filename"], change["status"], change["patch"]
            triplet_commit.append((f"COMMIT:{sha}", status.upper(), f"FILE:{filename}"))
            
            for line in patch.splitlines():
                if line.startswith("@@"):
                    name = line.split("@@")[2].strip()
                    if name.startswith("class"):
                        name = name[6:].strip()
                        if "(" in name:
                            name = name.split("(")[0].strip()
                        else:
                            name = name.split(":")[0].strip()
                        triplet_commit.append((f"COMMIT:{sha}", "MODIFIED", f"CLASS:{filename}:{name}"))

                    elif name.startswith("def"):
                        name = name[4:].strip()
                        if "(" in name:
                            name = name.split("(")[0].strip()
                        else:
                            name = name.split(":")[0].strip()
                        triplet_commit.append((f"COMMIT:{sha}", "MODIFIED", f"FUNC:{filename}:{name}"))

                    elif name.startswith("async def"):
                        name = name[10:].strip()
                        if "(" in name:
                            name = name.split("(")[0].strip()
                        else:
                            name = name.split(":")[0].strip()
                        triplet_commit.append((f"COMMIT:{sha}", "MODIFIED", f"FUNC:{filename}:{name}"))
    
    return remove_duplicate(triplet_commit)

### Load Dataset ###
name_data = args.d.strip().split("/")[1]
with open(f"./../data/{name_data}--{args.s}/data.pkl", "rb") as f:
    data = pickle.load(f)
### Load Dataset ###

### Load Models ###
auth = Auth.Token("GITHUB_TOKEN")
g = Github(auth=auth, pool_size=args.w)

nomicEmbedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cuda")

sem = asyncio.Semaphore(args.w)
client = AsyncOpenAI(api_key="OPENAI_API_KEY")

async def generate_summary_issue(text_issue):
    tasks = []
    for issue in text_issue:
        if text_issue[issue]:
            prompt_system = PROMPT_SYSTEM_SUMMARY_IP
            prompt_user = PROMPT_USER_SUMMARY_IP.format(content=text_issue[issue][:320000].strip())
            tasks.append(generate(issue, prompt_system, prompt_user))

    results = await tqdm.gather(*tasks)

    summary_issue = dict()
    for issue, response in results:
        summary = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        summary_issue[issue] = {"summary": summary, "raw_output": response}
    return summary_issue

async def generate_summary_commit(text_commit):
    tasks = []
    for sha in text_commit:
        if text_commit[sha]:
            prompt_system = PROMPT_SYSTEM_SUMMARY_C
            prompt_user = PROMPT_USER_SUMMARY_C.format(content=text_commit[sha][:320000].strip())
            tasks.append(generate(sha, prompt_system, prompt_user))

    results = await tqdm.gather(*tasks)

    summary_commit = dict()
    for sha, response in results:
        summary = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        summary_commit[sha] = {"summary": summary, "raw_output": response}
    return summary_commit

def parse_entities(raw_output, set_issue, set_commit):
    entities = set()
    for line in raw_output.splitlines():
        line = line.strip()

        if line.startswith("###"):
            str_type = line[3:].upper().strip()

        elif line:
            if line.startswith("-"):
                line = line[1:].strip()
            if not line:
                continue
            
            entity = line.strip()

            # Issue
            if line.upper().startswith("ISSUE#") or line.upper().startswith("ISSUE:") or line.upper().startswith("ISSUE_"):
                entity = f"ISSUE:{line[6:].strip()}"
                if entity not in set_issue:
                    continue
            # PR
            elif line.upper().startswith("PR#") or line.upper().startswith("PR:") or line.upper().startswith("PR_"):
                entity = f"PR:{line[3:].strip()}"
                if entity not in set_issue:
                    continue
            # Commit
            elif line.upper().startswith("COMMIT#") or line.upper().startswith("COMMIT:") or line.upper().startswith("COMMIT_"):
                entity = f"COMMIT:{line[7:].strip()[:6]}"
                if len(entity) < 13:
                    continue
                if entity not in set_commit:
                    continue

            entities.add(entity)
    
    return sorted(entities)

async def extract_entity(summaries, logger):
    tasks = []
    for target in summaries:
        if summaries[target]:
            prompt_system = PROMPT_SYSTEM_ENTITY
            prompt_user = PROMPT_USER_ENTITY.format(target=f"[{target}] {summaries[target][0]}", summary=summaries[target][1])
            tasks.append(generate(target, prompt_system, prompt_user))

    results = await tqdm.gather(*tasks)

    set_issue = set([item for item in summaries if (item.startswith("ISSUE:") or item.startswith("PR:"))])
    set_commit = set([item for item in summaries if item.startswith("COMMIT:")])
    
    extracted_entities, all_entities = dict(), []
    for target, response in results:
        raw_output = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        output = parse_entities(raw_output, set_issue, set_commit)

        if target not in output:
            output.append(target)
        
        all_entities += output

        extracted_entities[target] = {"entities": sorted(output), "raw_output": response}
    logger.info(f"Extracted {len(remove_duplicate(all_entities))} entities.")

    return extracted_entities

def format_entities(entities):
    if not entities:
        return ""

    return "\n".join(f"- {ent}" for ent in entities).strip()

def parse_triplets(output):
    triplets = []
    for line in output.splitlines():
        line = line.strip()
        if len(line.split("<|#|>")) != 3:
            continue
        h, r, t = line.split("<|#|>")
        triplets.append((h.strip(), r.strip(), t.strip()))
    return remove_duplicate(triplets)

async def extract_semantic_triplets(contents, logger):
    tasks = []
    for target in contents:
        if contents[target]:
            prompt_system = PROMPT_SYSTEM_TRIPLET
            prompt_user = PROMPT_USER_TRIPLET.format(target=f"[{target}] {contents[target][0]}", summary=contents[target][1], entities=format_entities(contents[target][2]))
            tasks.append(generate(target, prompt_system, prompt_user))

    results = await tqdm.gather(*tasks)

    extracted_triplets, all_triplets = dict(), []
    for target, response in results:
        raw_output = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        output = parse_triplets(raw_output)
        all_triplets += output

        extracted_triplets[target] = {"triplets": output, "raw_output": response}
    logger.info(f"Extracted {len(remove_duplicate(all_triplets))} triplets.")

    return extracted_triplets

async def generate(identifier, prompt_system, prompt_user):
    async with sem:
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        
        response = await client.chat.completions.create(
            model=args.m,
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.0
        )
        
        return identifier, response
### Load Models ###

### Logging ###
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

os.makedirs(f"./logs/{name_data}", exist_ok=True)
file_handler = logging.FileHandler(f"./logs/{name_data}/extract_kg.log", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
### Logging ###

def convert_timestamp(x):
    timestamp_ms = int(x) / 1000.0
    dt_object = datetime.fromtimestamp(timestamp_ms, tz=timezone.utc)
    iso_string = dt_object.isoformat()
    return iso_string

async def main():
    for idx in data:
        instance = data[idx]

        if os.path.exists(f"./res/{instance['instance_id']}/triplets.pkl"):
            continue
            
        reponame = instance["repo"]
        repo = g.get_repo(reponame)

        str_inst = f"### Extracting Knowledge Graph for {instance['instance_id']} ({idx}/{len(data)}) ###"
        logger.info("#" * len(str_inst))
        logger.info(str_inst)
        logger.info("#" * len(str_inst))

        if os.path.exists(f"./res/{instance['instance_id']}/triplet_schema_seed_{args.n}_hop_{args.hop}.txt"):
            with open(f"./res/{instance['instance_id']}/triplet_schema_seed_{args.n}_hop_{args.hop}.txt", "r") as f:
                triplets_schema = [tuple(line.strip().split("\t")) for line in f.readlines()]
            with open(f"./res/{instance['instance_id']}/dict_issue.json", "r") as f:
                dict_issue = json.load(f)
            with open(f"./res/{instance['instance_id']}/dict_commit.json", "r") as f:
                dict_commit = json.load(f)
            logger.info(f"Loaded Schematic Knowledge Graph for {instance['instance_id']}...")
        else:
            ### Preparation ###
            problem_statement = normalize_linesplit(instance.get("problem_statement"))
            problem_lines = set(line.strip() for line in problem_statement.splitlines())
            ### Preparation ###

            ### Load Issue List ###
            all_issues = []
            with open(f"./issues/{reponame.replace('/', '__')}.jsonl", "r") as f:
                for line in f:
                    all_issues.append(json.loads(line.strip()))
            ### Load Issue List ###

            ### Filter Issues ###
            if name_data == "SWE-bench-Live":
                issue_numbers = [(int(instance["instance_id"].split("-")[-1]), instance["created_at"] + "Z")]
                check = [(int(instance["instance_id"].split("-")[-1]), instance["created_at"] + "Z", "Question")]
            else:
                created_at = convert_timestamp(instance["created_at"])
                issue_numbers = [(int(instance["instance_id"].split("-")[-1]), created_at)]
                check = [(int(instance["instance_id"].split("-")[-1]), created_at, "Question")]

            for issue in all_issues:
                number = issue.get("number")
                title = issue.get("title")
                body = issue.get("body")
                created_at = issue.get("created_at")
                
                included = True
                if title:
                    if title.strip() not in problem_lines:
                        included = False
                if body:
                    body_lines = set(line.strip() for line in body.splitlines())
                    for line in body_lines:
                        if line not in problem_lines:
                            included = False

                if instance["instance_id"] == "streamlink__streamlink-6361" and number != 6360: # Issue named "."
                    included = False

                if instance["instance_id"] == "huggingface__diffusers-10115" and number != 10111: # Issue named "."
                    included = False

                if instance["instance_id"] == "mlflow__mlflow-10923" and number != 10891: # Issues with the same name
                    included = False

                if included:
                    issue_numbers.append((number, created_at))
                    check.append((number, created_at, title))

            if instance["instance_id"] == "kubernetes-client__python-2303": # Missing issue during retrieval
                issue_numbers.append((2298, "2024-10-10T13:12:32+00:00"))

            if instance["instance_id"] == "django__django-16421": # Django issues are treated on the external page.
                issue_numbers.append((30686, "2019-08-06T06:21:59-05:00"))

            if instance["instance_id"] == "django__django-18616": # Django issues are treated on the external page.
                issue_numbers.append((35784, "2024-09-24T02:16:08-05:00"))

            if instance["instance_id"] == "django__django-18435": # Django issues are treated on the external page.
                issue_numbers.append((35591, "2024-07-11T02:31:27-05:00"))

            if instance["instance_id"] == "django__django-19043": # Django issues are treated on the external page.
                issue_numbers.append((35521, "2024-06-12T06:17:10-05:00"))

            if instance["instance_id"] == "django__django-19009": # Django issues are treated on the external page.
                issue_numbers.append((36058, "2025-01-02T07:28:46-06:00"))

            if instance["instance_id"] == "django__django-13134": # Django issues are treated on the external page.
                issue_numbers.append((31741, "2020-06-27T16:32:43-05:00"))

            if instance["instance_id"] == "django__django-5605": # Django issues are treated on the external page.
                issue_numbers.append((20869, "2013-08-06T11:05:16-05:00"))

            if instance["instance_id"] == "django__django-18158": # Django issues are treated on the external page.
                issue_numbers.append((35515, "2024-06-09T09:08:02-05:00"))

            if instance["instance_id"] == "django__django-6478": # Django issues are treated on the external page.
                issue_numbers.append((27149, "2016-08-30T02:50:04-05:00"))

            if instance["instance_id"] == "django__django-18986": # Django issues are treated on the external page.
                issue_numbers.append((36052, "2025-01-01T17:08:41-06:00"))

            if instance["instance_id"] == "django__django-18941": # Django issues are treated on the external page.
                issue_numbers.append((36012, "2024-12-15T16:47:43-06:00"))

            if instance["instance_id"] == "django__django-18925": # Django issues are treated on the external page.
                issue_numbers.append((36007, "2024-12-12T16:43:04-06:00"))

            if instance["instance_id"] == "django__django-18911": # Django issues are treated on the external page.
                issue_numbers.append((35969, "2024-12-03T22:53:09-06:00"))

            if instance["instance_id"] == "django__django-18906": # Django issues are treated on the external page.
                issue_numbers.append((35988, "2024-12-09T05:11:35-06:00"))

            if instance["instance_id"] == "django__django-18905": # Django issues are treated on the external page.
                issue_numbers.append((35987, "2024-12-09T04:35:59-06:00"))

            if instance["instance_id"] == "django__django-18888": # Django issues are treated on the external page.
                issue_numbers.append((35973, "2024-12-04T14:50:24-06:00"))

            if instance["instance_id"] == "django__django-18854": # Django issues are treated on the external page.
                issue_numbers.append((35944, "2024-11-26T15:08:21-06:00"))

            if instance["instance_id"] == "django__django-18850": # Django issues are treated on the external page.
                issue_numbers.append((35493, "2024-06-02T10:17:21-05:00"))

            if instance["instance_id"] == "django__django-18846": # Django issues are treated on the external page.
                issue_numbers.append((35935, "2024-11-24T16:06:53-06:00"))

            if instance["instance_id"] == "django__django-18820": # Django issues are treated on the external page.
                issue_numbers.append((35913, "2024-11-16T01:19:29-06:00"))

            if instance["instance_id"] == "django__django-18796": # Django issues are treated on the external page.
                issue_numbers.append((35903, "2024-11-11T08:28:13-06:00"))

            if instance["instance_id"] == "django__django-18795": # Django issues are treated on the external page.
                issue_numbers.append((35815, "2024-10-04T08:37:27-05:00"))

            if instance["instance_id"] == "django__django-18785": # Django issues are treated on the external page.
                issue_numbers.append((35897, "2024-11-08T02:47:56-06:00"))

            if instance["instance_id"] == "django__django-18752": # Django issues are treated on the external page.
                issue_numbers.append((35882, "2024-11-02T02:44:19-05:00"))

            if instance["instance_id"] == "django__django-18653": # Django issues are treated on the external page.
                issue_numbers.append((35789, "2024-09-26T04:44:43-05:00"))

            if instance["instance_id"] == "django__django-18508": # Django issues are treated on the external page.
                issue_numbers.append((35700, "2024-08-22T05:57:37-05:00"))

            if instance["instance_id"] == "django__django-18470": # Django issues are treated on the external page.
                issue_numbers.append((35669, "2024-08-10T01:47:11-05:00"))

            if instance["instance_id"] == "django__django-18105": # Django issues are treated on the external page.
                issue_numbers.append((35408, "2024-04-26T05:05:01-05:00"))

            if instance["instance_id"] == "django__django-18104": # Django issues are treated on the external page.
                issue_numbers.append((35407, "2024-04-26T04:13:26-05:00"))

            if instance["instance_id"] == "django__django-18059": # Django issues are treated on the external page.
                issue_numbers.append((35364, "2024-04-09T08:38:10-05:00"))

            if instance["instance_id"] == "django__django-17984": # Django issues are treated on the external page.
                issue_numbers.append((35309, "2024-03-15T16:54:19-05:00"))

            if instance["instance_id"] == "django__django-17904": # Django issues are treated on the external page.
                issue_numbers.append((35250, "2024-02-24T13:16:25-06:00"))

            if instance["instance_id"] == "django__django-17885": # Django issues are treated on the external page.
                issue_numbers.append((26001, "2015-12-29T07:43:35-06:00"))

            if instance["instance_id"] == "django__django-17874": # Django issues are treated on the external page.
                issue_numbers.append((35232, "2024-02-18T15:49:30-06:00"))

            if instance["instance_id"] == "vllm-project__vllm-10778": # Missing issue during retrieval
                issue_numbers.append((10772, "2024-11-29T10:14:52Z"))
                issue_numbers.append((10764, "2024-11-29T05:40:18Z"))

            if instance["instance_id"] == "vllm-project__vllm-10620": # Missing issue during retrieval
                issue_numbers.append((10619, "2024-11-25T06:23:52Z"))

            if instance["instance_id"] == "django__django-18654": # Django issues are treated on the external page.
                issue_numbers.append((35807, "2024-10-02T16:26:44-05:00"))

            for number, created_at, title in check:
                logger.info(f"{number} {created_at} {title}")

            times = [datetime.fromisoformat(created_at.replace("Z", "+00:00")) for _, created_at in issue_numbers]
            dt = min(times)
            logger.info(f"Cutoff Date: {dt}")
            ### Filter Issues ###

            ### Retrive Relevant Issues ###
            candidates = []
            number_title = dict()
            for issue in all_issues:
                number = issue.get("number")
                title = issue.get("title")
                created_at = issue.get("created_at")
                if not created_at:
                    continue

                if datetime.fromisoformat(created_at.replace("Z", "+00:00")) < dt:
                    candidates.append(number)
                    number_title[number] = title.strip()

            titles = [number_title[number] for number in candidates]

            embedding_target = nomicEmbedder.encode("search_query: " + problem_statement)
            embedding_issues = nomicEmbedder.encode(["search_document: " + normalize_linesplit(text) for text in titles])

            similarity = np.sum(embedding_target * embedding_issues, -1)
            retrieve_idx = np.argsort(-similarity)[:args.n].tolist()
            ### Retrive Relevant Issues ###

            ### Extract Schematic KG ###
            _, triplets_schema, dict_issue = await expand_kg(repo, dt, [set(), set(candidates[idx] for idx in retrieve_idx)], args.hop, set(), dict(), logger)
            triplets_schema = sorted(triplets_schema)
            triplets_schema = [list(triplet) for triplet in triplets_schema]

            list_sha = []
            for h, r, t in triplets_schema:
                if t.startswith("COMMIT:"):
                    list_sha.append(t[7:])
                list_sha = sorted(set(list_sha))
            dict_commit = await gather_commits(repo, list_sha, logger)

            for i in range(len(triplets_schema)):
                if triplets_schema[i][0].startswith("COMMIT:"):
                    triplets_schema[i][0] = triplets_schema[i][0][:13]
                if triplets_schema[i][2].startswith("COMMIT:"):
                    triplets_schema[i][2] = triplets_schema[i][2][:13]
            triplets_schema = [tuple(triplet) for triplet in triplets_schema]

            triplets_commit = extract_commit_triplets(dict_commit)
            triplets_schema += triplets_commit
            triplets_schema = remove_duplicate(triplets_schema)

            os.makedirs(f"./res/{instance['instance_id']}/", exist_ok=True)
            with open(f"./res/{instance['instance_id']}/dict_issue.json", "w") as f:
                json.dump(dict_issue, f, indent=4, ensure_ascii=False)

            with open(f"./res/{instance['instance_id']}/dict_commit.json", "w") as f:
                json.dump(dict_commit, f, indent=4, ensure_ascii=False)

            with open(f"./res/{instance['instance_id']}/triplet_schema_seed_{args.n}_hop_{args.hop}.txt", "w") as f:
                for h, r, t in sorted(triplets_schema):
                    f.write(f"{h}\t{r}\t{t}\n")
            ### Extract Schematic KG ###
            logger.info(f"Extracted {len(triplets_schema)} triplets")

        if os.path.exists(f"./res/{instance['instance_id']}/text_issue.json"):
            with open(f"./res/{instance['instance_id']}/text_issue.json", "r") as f:
                text_issue = json.load(f)
            logger.info(f"Loaded Issue/PR text for {instance['instance_id']}...")
        else:
            text_issue = {issue: format_issue(issue, dict_issue[issue]) for issue in dict_issue}
            with open(f"./res/{instance['instance_id']}/text_issue.json", "w") as f:
                json.dump(text_issue, f, indent=4, ensure_ascii=False)

        if os.path.exists(f"./res/{instance['instance_id']}/text_commit.json"):
            with open(f"./res/{instance['instance_id']}/text_commit.json", "r") as f:
                text_commit = json.load(f)
            logger.info(f"Loaded Commit text for {instance['instance_id']}...")
        else:
            text_commit = {sha: format_commit(sha, dict_commit[sha]) for sha in dict_commit}
            with open(f"./res/{instance['instance_id']}/text_commit.json", "w") as f:
                json.dump(text_commit, f, indent=4, ensure_ascii=False)

        if os.path.exists(f"./res/{instance['instance_id']}/summary_issue.pkl"):
            with open(f"./res/{instance['instance_id']}/summary_issue.pkl", "rb") as f:
                summary_issue = pickle.load(f)
            logger.info(f"Loaded Issue/PR summary for {instance['instance_id']}...")
        else:
            summary_issue = await generate_summary_issue(text_issue)
            with open(f"./res/{instance['instance_id']}/summary_issue.pkl", "wb") as f:
                pickle.dump(summary_issue, f)

        if os.path.exists(f"./res/{instance['instance_id']}/summary_commit.pkl"):
            with open(f"./res/{instance['instance_id']}/summary_commit.pkl", "rb") as f:
                summary_commit = pickle.load(f)
            logger.info(f"Loaded Commit summary for {instance['instance_id']}...")
        else:
            summary_commit = await generate_summary_commit(text_commit)
            with open(f"./res/{instance['instance_id']}/summary_commit.pkl", "wb") as f:
                pickle.dump(summary_commit, f)
        
        if os.path.exists(f"./res/{instance['instance_id']}/entity.pkl"):
            with open(f"./res/{instance['instance_id']}/entity.pkl", "rb") as f:
                entity = pickle.load(f)
            logger.info(f"Loaded Entities for {instance['instance_id']}...")
        else:
            summaries = dict()
            for issue in summary_issue:
                if not summary_issue[issue]["summary"]:
                    continue

                if dict_issue[issue]["is_pr"]:
                    target = f"PR:{issue}"
                    title = f"{dict_issue[issue]['title']}".strip()
                else:
                    target = f"ISSUE:{issue}"
                    title = f"{dict_issue[issue]['title']}".strip()
                summaries[target] = (title, summary_issue[issue]["summary"])
            for sha in summary_commit:
                if not summary_commit[sha]["summary"]:
                    continue
                target = f"COMMIT:{sha}"
                if dict_commit[sha]:
                    title = f"{dict_commit[sha]['message']}".strip().splitlines()[0].strip()
                else:
                    title = ""
                summaries[target] = (title, summary_commit[sha]["summary"])

            for issue in dict_issue:
                if not dict_issue[issue]:
                    continue
                if dict_issue[issue]["is_pr"] and (f"PR:{issue}" not in summaries):
                    summaries[f"PR:{issue}"] = None
                elif (not dict_issue[issue]["is_pr"]) and (f"ISSUE:{issue}" not in summaries):
                    summaries[f"ISSUE:{issue}"] = None
            for sha in dict_commit:
                if not dict_commit[sha]:
                    continue
                if f"COMMIT:{sha}" not in summaries:
                    summaries[f"COMMIT:{sha}"] = None

            entity = await extract_entity(summaries, logger)
            with open(f"./res/{instance['instance_id']}/entity.pkl", "wb") as f:
                pickle.dump(entity, f)

        if os.path.exists(f"./res/{instance['instance_id']}/triplets.pkl"):
            with open(f"./res/{instance['instance_id']}/triplets.pkl", "rb") as f:
                triplets = pickle.load(f)
            logger.info(f"Loaded Triplets for {instance['instance_id']}...")
        else:
            contents = dict()
            for issue in summary_issue:
                if not summary_issue[issue]:
                    continue
                if dict_issue[issue]["is_pr"]:
                    target = f"PR:{issue}"
                    title = f"{dict_issue[issue]['title']}".strip()
                else:
                    target = f"ISSUE:{issue}"
                    title = f"{dict_issue[issue]['title']}".strip()
                if entity.get(target):
                    contents[target] = (title, summary_issue[issue]["summary"], entity[target]["entities"])

            for sha in summary_commit:
                if not summary_commit[sha]:
                    continue
                target = f"COMMIT:{sha}"
                if dict_commit[sha]:
                    title = f"{dict_commit[sha]['message']}".strip().splitlines()[0].strip()
                else:
                    title = ""
                if entity.get(target):
                    contents[target] = (title, summary_commit[sha]["summary"], entity[target]["entities"])
            
            triplets = await extract_semantic_triplets(contents, logger)
            with open(f"./res/{instance['instance_id']}/triplets.pkl", "wb") as f:
                pickle.dump(triplets, f)

if __name__ == "__main__":
    asyncio.run(main())