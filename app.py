import json
import requests
from flask import Flask, render_template, request, Response, stream_with_context

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"  # change to whatever model you have pulled


BEGIN_DOC_MARKER = r"\begin{document}"
END_DOC_MARKER = r"\end{document}"


def split_latex(full_tex: str):
    """Split LaTeX into (preamble, body). Preamble includes \\begin{document}."""
    idx = full_tex.find(BEGIN_DOC_MARKER)
    if idx == -1:
        return "", full_tex
    split_at = idx + len(BEGIN_DOC_MARKER)
    preamble = full_tex[:split_at]
    body = full_tex[split_at:]
    end_idx = body.rfind(END_DOC_MARKER)
    if end_idx != -1:
        body = body[:end_idx]
    return preamble, body


def build_prompt(body: str, job_description: str, skills: str) -> str:
    return f"""You are a senior technical resume writer. A real person wrote this resume and needs it tailored for a specific job posting. Your job is to rewrite bullet points and text so the resume reads like a strong human-written application — not an AI output.

CONTEXT:
- You are receiving ONLY the document body (between \\begin{{document}} and \\end{{document}}).
- The LaTeX preamble and custom commands are handled separately. Do NOT output them.

WHAT YOU MUST CHANGE:
1. SKILLS SECTION — THIS IS YOUR TOP PRIORITY:
   - You will find a "Technical Skills" section with categories like "Programming \\& Scripting", "Web, Frontend \\& Backend", "Cloud, DevOps \\& Tools".
   - You MUST insert EVERY skill from the "SKILLS TO ADD" list below into the appropriate category.
   - Append each new skill to the end of the matching category's comma-separated list.
   - If a skill is already listed, do NOT duplicate it.
   - If a skill doesn't fit any existing category, add it to the closest one.
   - Example: if SKILLS TO ADD contains "Terraform, Go" — add "Terraform" to "Cloud, DevOps \\& Tools" and "Go" to "Programming \\& Scripting".
2. REWRITE bullet points to directly address the job description. Extract the key responsibilities and required qualifications from the job posting, then reshape each relevant bullet to demonstrate that the candidate has done exactly that work. Use the same terminology the job posting uses, but weave it naturally into the candidate's actual experience.
3. PROFESSIONAL SUMMARY: Rewrite it to mirror the job's core requirements. Lead with the most relevant qualification. Keep it 2–3 sentences max, confident but not generic.
4. REORDER bullet points within each job/project so the most job-relevant ones come first.

WRITING STYLE — THIS IS CRITICAL:
- Write like a real person, not a template. Vary sentence structure and length.
- Be specific: use numbers, tools, outcomes. "Reduced API response time by 40ms using Redis caching" beats "Improved system performance through optimization."
- Never chain more than 2 buzzwords together. Break up jargon with concrete details.
- Don't start every bullet with "Developed" or "Implemented" — vary your verbs: built, designed, shipped, migrated, cut, drove, led, automated, debugged, profiled, etc.
- Keep the candidate's voice. If they write casually, stay casual. If formal, stay formal.
- Do NOT add experience or achievements the candidate doesn't already have. Only rephrase and reframe existing work.

LATEX RULES — DO NOT BREAK THESE:
1. Output ONLY the modified body. No \\begin{{document}}, no \\end{{document}}, no preamble.
2. Do NOT modify LaTeX structural commands: \\section, \\resumeSubheading, \\resumeItem, \\resumeProjectHeading, \\vspace, \\resumeSubHeadingListStart/End, \\resumeItemListStart/End, \\begin{{center}}, \\begin{{tabular*}}, \\begin{{itemize}}, etc.
3. ONLY change the text content inside these commands — the arguments in curly braces.
4. Preserve ALL \\vspace values, spacing, and formatting exactly.
5. Preserve ALL LaTeX escape sequences exactly: \\&, \\%, \\$, \\#, \\_, \\textbf, \\textit, \\href, \\underline, $\\vert$, $\\boldsymbol$, etc. Never strip backslashes from these.
6. Add % MODIFIED at the end of every line you change.
7. No explanations, no markdown fences, no commentary — just the LaTeX body.

--- RESUME BODY ---
{body}

--- JOB DESCRIPTION ---
{job_description}

--- SKILLS TO ADD ---
{skills}

Output the modified body now:"""


@app.route("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL)


@app.route("/models")
def get_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return {"models": models, "status": "ok"}
    except Exception as e:
        return {"models": [], "status": "error", "error": str(e)}, 500


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    latex_resume = data.get("latex_resume", "").strip()
    job_description = data.get("job_description", "").strip()
    skills = data.get("skills", "").strip()
    model = data.get("model", DEFAULT_MODEL).strip()

    if not latex_resume or not job_description:
        return {"error": "Resume and job description are required."}, 400

    preamble, body = split_latex(latex_resume)
    prompt = build_prompt(body, job_description, skills)

    def stream_response():
        yield f"data: {json.dumps({'preamble': preamble})}\n\n"
        try:
            with requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": True},
                stream=True,
                timeout=300,
            ) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Ollama returned {r.status_code}'})}\n\n"
                    return

                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        done = chunk.get("done", False)
                        yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                        if done:
                            break
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama. Make sure it is running on port 11434.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(stream_response()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)
