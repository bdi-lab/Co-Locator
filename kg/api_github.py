from github import GithubException
from github.GithubException import UnknownObjectException
from tqdm import tqdm

import requests
import json
import os
import re

def normalize_linesplit(text):
    return "\n".join(text.splitlines())

def convert_issue(issue):
    if issue.pull_request:
        text = f"PR:{issue.number}"
    else:
        text = f"ISSUE:{issue.number}"
    return text

def process_timeline(issue, timeline, dt):
    type_timeline = timeline.event
    if type_timeline == "labeled":
        return (issue, "LABELED", "LABEL:" + timeline.raw_data['label']['name'])
    elif type_timeline == "unlabeled":
        return (issue, "UNLABELED", "LABEL:" + timeline.raw_data['label']['name'])
    elif type_timeline == "milestoned":
        return (issue, "MILESTONED", "MILESTONE:" + timeline.raw_data['milestone']['title'])
    elif type_timeline == "demilestoned":
        return (issue, "DEMILESTONED", "MILESTONE:" + timeline.raw_data['milestone']['title'])
    elif type_timeline == "merged":
        return (issue, "MERGED", "COMMIT:" + timeline.commit_id)
    elif type_timeline == "referenced":
        return (issue, "REFERENCED", "COMMIT:" + timeline.commit_id)
    elif type_timeline == "committed":
        return (issue, "COMMITTED", "COMMIT:" + timeline.raw_data['sha'])
    elif type_timeline == "cross-referenced":
        return (issue, "CROSS-REFERENCED", convert_issue(timeline.source.issue))
    else:
        return None

def extract_triplets(repo, number, dt, logger):
    try:
        issue = repo.get_issue(number=int(number))
        # logger.info(f"Extracting schema triplets from [{number}] {issue.title}")
    except UnknownObjectException:
        # logger.info(f"ERROR: 404 NOT FOUND")
        return [], None
    except:
        return [], None

    dict_content = dict()
    is_pr = True if issue.pull_request else False
    dict_content["is_pr"] = is_pr
    dict_content["title"] = issue.title.strip()
    dict_content["body"] = normalize_linesplit(issue.body.strip()) if issue.body else None
    dict_content["created_at"] = issue.created_at.isoformat() if issue.created_at else None
    dict_content["user"] = issue.user.login if issue.user.login else "UNKNOWN"
    dict_content["comments"] = []

    triplets = []
    curr_dt = None
    for timeline in list(issue.get_timeline())[::-1]:
        if timeline.created_at:
            curr_dt = timeline.created_at
        if (not curr_dt) or (curr_dt >= dt):
            continue

        triplet = process_timeline(convert_issue(issue), timeline, dt)
        if triplet:
            triplets.append(triplet)

        if timeline.event == "commented":
            body = normalize_linesplit(timeline.body.strip()) if timeline.body else None
            created_at = timeline.created_at.isoformat() if timeline.created_at else None
            user = timeline.actor.login if timeline.actor.login else "UNKNOWN"
            dict_content["comments"].append({"body": body, "created_at": created_at, "user": user})

    return triplets, dict_content

def extract_commit(repo, sha, logger):
    try:
        commit = repo.get_commit(sha=sha)
        # logger.info(f"Extracting commits with sha: {sha}")
    except UnknownObjectException:
        # logger.info(f"ERROR: 404 NOT FOUND")
        return None
    except GithubException as e:
        # logger.info(f"ERROR CODE: 422")
        if e.status == 422:
            return None

    dict_content = dict()
    dict_content["sha"] = sha[:6]
    dict_content["message"] = commit.commit.message.strip() if commit.commit.message else None
    dict_content["changes"] = []
    for file in commit.files:
        filename = file.filename
        status = file.status
        patch = file.patch if file.patch else None
        dict_content["changes"].append({"filename": filename, "status": status, "patch": patch})

    return dict_content

TEMPLATE_ISSUE = """
[ISSUE:{number}] by {user}

Title: {title}

{body}

{comments}
""".strip()

TEMPLATE_PR = """
[PR:{number}] by {user}

Title: {title}

{body}
{comments}
""".strip()

TEMPLATE_COMMENT = """
--------------------------------------------------------------------------------
[COMMENT] by {user}

{body}
""".strip()

def format_issue(number, dict_i):
    list_comments = [TEMPLATE_COMMENT.format(user=comment.get("user"), body=comment.get("body")).strip() for comment in dict_i["comments"]]
    if dict_i["is_pr"]:
        text = TEMPLATE_PR.format(number=number, user=dict_i.get("user"), title=dict_i.get("title"), body=dict_i.get("body"), comments="\n".join(list_comments))
    else:
        text = TEMPLATE_ISSUE.format(number=number, user=dict_i.get("user"), title=dict_i.get("title"), body=dict_i.get("body"), comments="\n".join(list_comments))
    
    return text.strip()

TEMPLATE_COMMIT = """
[COMMIT:{sha}]

{message}
{changes}
""".strip()

TEMPLATE_CHANGE = """
--------------------------------------------------------------------------------
[{status}] {filename}

{patch}
""".strip()

def is_test_file(file):
    word_list = re.split(r" |_|\/", file.lower())
    return any([word.startswith("test") for word in word_list])

def format_commit(sha, dict_c):
    if dict_c:
        py_changes = []
        for change in dict_c.get("changes"):
            if change.get("filename").endswith(".py") and not is_test_file(change.get("filename")):
                py_changes.append(change)

        list_changes = [TEMPLATE_CHANGE.format(status=change.get("status"), filename=change.get("filename"), patch=change.get("patch")).strip() for change in py_changes]
        text = TEMPLATE_COMMIT.format(sha=sha, message=dict_c.get("message"), changes="\n".join(list_changes))
        
        return text.strip()

    else:
        return ""