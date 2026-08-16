
import os
import requests

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from langserve import add_routes
from pypdf import PdfReader

from ddgs import DDGS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Job Ready LangChain Agent",
    version="1.0"
)


# =========================================================
# LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)


# =========================================================
# TOOL 1 — JOB SEARCH
# =========================================================

@tool
def job_search(role: str) -> str:
    """Search for relevant job opportunities in India."""

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


# =========================================================
# TOOL 2 — SKILL GAP
# =========================================================

@tool
def skill_gap(resume_text: str, target_role: str) -> str:
    """Analyze the student's skill gaps for the target role."""

    prompt = f"""
You are a career skill-gap analyzer.

Student Resume:
{resume_text}

Target Role:
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


# =========================================================
# TOOL 3 — PROJECT IDEAS
# =========================================================

@tool
def project_ideas(target_role: str) -> str:
    """Suggest useful portfolio projects for the target role."""

    prompt = f"""
You are a project recommendation assistant.

Target Role:
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


# =========================================================
# TOOL 4 — GITHUB CHECK
# =========================================================

@tool
def github_check(username: str) -> str:
    """Check a GitHub user's public profile and repositories."""

    profile_url = f"https://api.github.com/users/{username}"

    repos_url = (
        f"https://api.github.com/users/{username}/repos"
        "?sort=updated&per_page=5"
    )

    profile_response = requests.get(profile_url, timeout=10)

    if profile_response.status_code != 200:
        return f"GitHub user '{username}' was not found."

    profile = profile_response.json()

    repos_response = requests.get(repos_url, timeout=10)

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


# =========================================================
# AGENT
# =========================================================

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

Your job is to analyze a student's career readiness.

You receive:
1. Resume
2. Target role
3. GitHub username

You have four tools:

1. job_search
   Search for relevant job opportunities.

2. skill_gap
   Compare the student's resume skills with the target role.

3. project_ideas
   Suggest useful portfolio projects.

4. github_check
   Analyze the student's public GitHub activity.

Use the appropriate tools to analyze the student.

You may call multiple tools.

Finally, combine the relevant information into ONE
clear Career Readiness Report containing:

1. Current Skills
2. Skill Gaps
3. GitHub Analysis
4. Recommended Projects
5. Relevant Job Opportunities
6. Final Recommendation

Do not simply describe the tools.
Actually use them when appropriate.
"""
)


# =========================================================
# LANGSERVE ROUTE
# =========================================================

add_routes(
    app,
    agent,
    path="/agent"
)


# =========================================================
# HOME PAGE — STUDENT UI
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html>
<head>
    <title>Job Ready Agent</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f7fb;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }

        h1 {
            margin-bottom: 5px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 25px;
        }

        label {
            display: block;
            margin-top: 18px;
            margin-bottom: 7px;
            font-weight: bold;
        }

        input {
            width: 100%;
            padding: 12px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 8px;
        }

        button {
            margin-top: 25px;
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        #status {
            margin-top: 20px;
            font-weight: bold;
        }

        #result {
            margin-top: 20px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 10px;
            white-space: pre-wrap;
            line-height: 1.5;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>🎯 Job Readiness Agent</h1>

    <p class="subtitle">
        Upload your resume and get an AI-powered career readiness analysis.
    </p>

    <form id="form">

        <label>Resume PDF</label>
        <input
            type="file"
            id="resume"
            accept=".pdf"
            required
        >

        <label>Target Job Role</label>
        <input
            type="text"
            id="role"
            placeholder="Example: AI Engineer"
            required
        >

        <label>GitHub Username</label>
        <input
            type="text"
            id="github"
            placeholder="Enter your GitHub username"
            required
        >

        <button type="submit">
            Analyze Career Readiness
        </button>

    </form>

    <div id="status"></div>

    <div id="result"></div>

</div>

<script>

document.getElementById("form").addEventListener("submit", async function(event) {

    event.preventDefault();

    const resume = document.getElementById("resume").files[0];
    const role = document.getElementById("role").value;
    const github = document.getElementById("github").value;

    const formData = new FormData();

    formData.append("resume", resume);
    formData.append("role", role);
    formData.append("github_username", github);

    document.getElementById("status").innerText =
        "⏳ Analyzing resume, GitHub and job readiness...";

    document.getElementById("result").innerText = "";

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Analysis failed");
        }

        document.getElementById("status").innerText =
            "✅ Analysis complete";

        document.getElementById("result").innerText =
            data.report;

    } catch (error) {

        document.getElementById("status").innerText =
            "❌ Error";

        document.getElementById("result").innerText =
            error.message;
    }

});

</script>

</body>
</html>
"""


# =========================================================
# ANALYZE ENDPOINT
# =========================================================

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_username: str = Form(...)
):

    # -----------------------------
    # Validate PDF
    # -----------------------------

    if not resume.filename.lower().endswith(".pdf"):
        return {
            "error": "Please upload a PDF resume."
        }


    # -----------------------------
    # Extract PDF text
    # -----------------------------

    contents = await resume.read()

    temp_path = "temp_resume.pdf"

    with open(temp_path, "wb") as f:
        f.write(contents)

    reader = PdfReader(temp_path)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text + "\n"


    # -----------------------------
    # Create Agent Request
    # -----------------------------

    prompt = f"""
Analyze this student's career readiness.

IMPORTANT:
The user has explicitly selected the target role below.
You MUST use this exact role throughout the entire report.
Do NOT change it, infer a different role, or replace it
with a role found in the resume.

TARGET ROLE:
{role}

GITHUB USERNAME:
{github_username}

RESUME:
{resume_text[:12000]}

Your tasks:

1. Analyze the student's skills against the EXACT target role.
2. Identify skill gaps specifically for the EXACT target role.
3. Analyze the student's GitHub activity.
4. Suggest projects specifically for the EXACT target role.
5. Find job opportunities relevant to the EXACT target role.
6. Produce one concise Career Readiness Report.

The report MUST begin with:

# Career Readiness Report

**Target Role:** {role}

Never replace "{role}" with another job title.

Use the available tools when necessary.
"""


    # -----------------------------
    # Invoke Agent
    # -----------------------------

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    # -----------------------------
    # Extract final response
    # -----------------------------

    final_message = result["messages"][-1]

    content = final_message.content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)

        report = "\n".join(text_parts)

    else:
        report = str(content)

    return {
        "target_role": role,
        "github_username": github_username,
        "report": report
    }
