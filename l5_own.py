import requests
from dotenv import load_dotenv
import time
import math
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import dotenv_values
from huggingface_hub import list_datasets
from huggingface_hub import create_repo
from huggingface_hub import Repository
from datasets import load_dataset,Features, Value
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
# response = requests.get(url)

# print(response.json())

config = dotenv_values(".env")

print(config)   

GITHUB_TOKEN = config.get("GITHUB_TOKEN")  # Copy your GitHub token here
headers = {"Authorization": f"token {GITHUB_TOKEN}"}



def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )
    
# fetch_issues()


# features = Features({
#        "field1": Value("string"),
#        "timestamp_field": Value("timestamp[s]"),
#        "timeline_url": Value("string"),
#        "performed_via_github_app": Value("null"),
#        "state_reason": Value("string"),
#        "draft": Value("bool"),
#        "pull_request": {
#            "url": Value("string"),
#            "html_url": Value("string"),
#            "diff_url": Value("string"),
#            "patch_url": Value("string"),
#            "merged_at": Value("timestamp[s]")
#        },
#        # 添加其他字段
#    })
issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")



sample = issues_dataset.shuffle(seed=666).select(range(3))

# Print out the URL and pull request entries
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")
    
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)
response.json()

def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


# Test our function works as expected
get_comments(2792)

issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)

issues_with_comments_dataset.to_json("issues-datasets-with-comments.jsonl")
all_datasets = list_datasets()

repo_url = create_repo(name="github-issues", repo_type="dataset")
repo = Repository(local_dir="github-issues", clone_from=repo_url)
repo.lfs_track("*.jsonl")
repo.push_to_hub()