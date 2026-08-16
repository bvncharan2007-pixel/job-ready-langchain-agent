
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from langserve import add_routes
from pypdf import PdfReader

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# API
# -----------------------------

app = FastAPI(
    title="Job Ready LangChain Agent",
    version="1.0"
)


# -----------------------------
# Model
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)


# -----------------------------
# Tools
# -----------------------------

from ddgs import DDGS
import requests
from langchain_core.tools import tool


@tool
def job_search(role: str) -> str:
    """Search the web for relevant job opportunities."""

    results = DDGS().text(
        f"{role} jobs India",
        max_results=5
    )

    if not results:
        return "No job results found."

    output = []

    for i, result in enumerate(results, 1):
        output.append(
            f"{i}. {result['title']}\n"
            f"Link: {result['href']}\n"
            f"{result.get('body', '')}"
        )

    return "\n\n".join(output)


@tool
def skill_gap(resume_text: str, target_role: str) -> str:
    """Analyze the student's skill gaps for the target role."""

    prompt = f"""
You are a career skill-gap analyzer.

Student resume:
{resume_text}

Target role:
{target_role}

Analyze the student's skills against the expected skills
for the target role.

Return:
1. Skills already present
2. Missing or weak skills
3. Top 3 skills to learn next

Keep the response concise and practical.
"""

    response = llm.invoke(prompt)

    return response.content


@tool
def project_ideas(target_role: str) -> str:
    """Suggest portfolio projects for the target role."""

    prompt = f"""
You are a project recommendation assistant.

Target role:
{target_role}

Suggest 3 practical projects that would strengthen
a student's portfolio.

For each project provide:
1. Project name
2. What it does
3. Technologies
4. Why it is useful

Keep the suggestions realistic for a student.
"""

    response = llm.invoke(prompt)

    return response.content


@tool
def github_check(username: str) -> str:
    """Check a GitHub user's public profile and repositories."""

    profile_url = f"https://api.github.com/users/{username}"
    repos_url = (
        f"https://api.github.com/users/{username}/repos"
        "?sort=updated&per_page=5"
    )

    profile_response = requests.get(profile_url)

    if profile_response.status_code != 200:
        return f"GitHub user '{username}' was not found."

    profile = profile_response.json()

    repos_response = requests.get(repos_url)

    if repos_response.status_code != 200:
        return "Unable to retrieve GitHub repository activity."

    repos = repos_response.json()

    output = [
        f"GitHub User: {profile.get('login')}",
        f"Public Repositories: {profile.get('public_repos')}",
        f"Followers: {profile.get('followers')}",
        "",
        "Recent Repository Activity:"
    ]

    for repo in repos:
        output.append(
            f"- {repo.get('name')} | "
            f"Stars: {repo.get('stargazers_count')} | "
            f"Language: {repo.get('language')} | "
            f"Updated: {repo.get('updated_at')}"
        )

    return "\n".join(output)


# -----------------------------
# Agent
# -----------------------------

tools = [
    job_search,
    skill_gap,
    project_ideas,
    github_check
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a Job Readiness Agent.

Analyze a student's career readiness using:

1. Resume
2. Target role
3. GitHub username

Available tools:

- job_search: Find relevant job opportunities.
- skill_gap: Analyze missing skills.
- project_ideas: Recommend portfolio projects.
- github_check: Analyze GitHub activity.

Choose appropriate tools based on the request.

You may call multiple tools.

Finally, synthesize all relevant information into
one concise career-readiness report.
"""
)


# -----------------------------
# LangServe route
# -----------------------------

add_routes(
    app,
    agent,
    path="/agent"
)


@app.get("/")
def home():
    return {
        "message": "Job Ready LangChain Agent is running"
    }
