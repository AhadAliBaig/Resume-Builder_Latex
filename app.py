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


@app.route("/extract-keywords", methods=["POST"])
def extract_keywords():
    data = request.get_json()
    job_description = data.get("job_description", "").strip()
    resume_text = data.get("resume_text", "").strip()
    model = data.get("model", DEFAULT_MODEL).strip()

    if not job_description:
        return {"error": "Job description is required."}, 400

    prompt = f"""Extract the key technical skills, tools, frameworks, programming languages, platforms, methodologies, and qualifications from this job description that an ATS (Applicant Tracking System) would scan for.

RULES:
- Return ONLY a valid JSON array of strings. No explanations, no markdown.
- Include specific tools/technologies (e.g. "Kubernetes", "React", "PostgreSQL"), not generic phrases.
- Include methodologies and practices (e.g. "Agile", "CI/CD", "TDD").
- Include soft skills only if explicitly required (e.g. "leadership", "mentoring").
- Keep each item short — 1 to 3 words max.
- Aim for 15–30 keywords. Cover all important ones.

Example output: ["Python", "React", "AWS", "CI/CD", "Docker", "Agile", "REST API", "PostgreSQL"]

Job Description:
{job_description}

JSON array:"""

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if r.status_code != 200:
            return {"error": f"Ollama returned {r.status_code}"}, 500

        raw = r.json().get("response", "").strip()

        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return {"error": "Model did not return valid JSON array.", "raw": raw}, 500

        keywords = json.loads(raw[start:end + 1])
        keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]

        resume_lower = resume_text.lower()
        results = []
        for kw in keywords:
            present = kw.lower() in resume_lower
            results.append({"keyword": kw, "present": present})

        present_count = sum(1 for r in results if r["present"])
        total = len(results)
        score = round((present_count / total) * 100) if total > 0 else 0

        return {
            "keywords": results,
            "match_score": score,
            "present_count": present_count,
            "total": total,
        }
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Ollama."}, 500
    except json.JSONDecodeError:
        return {"error": "Failed to parse keywords from model response.", "raw": raw}, 500
    except Exception as e:
        return {"error": str(e)}, 500


def build_cover_letter_prompt(resume_text: str, job_description: str) -> str:
    return f"""You are a professional cover letter writer. Write a one-page cover letter for a job application.

FORMAT — follow this structure exactly:
Line 1: Current month and year (e.g. "March 2026")
Line 2: Company/organization name (extract from job description)
Line 3: City, Province/State
Line 4: blank
Line 5: "Re: Cover Letter – [exact job title from the posting]"
Line 6: blank
Line 7: "Dear Hiring Manager,"
Line 8: blank
Lines 9-end: Three paragraphs, then sign-off.

PARAGRAPH STRUCTURE:
- Paragraph 1 (3-4 sentences): Who the candidate is (year, program, university), why they are excited about THIS specific role. Reference something specific about the company or role that connects to their interests. Do not be generic.
- Paragraph 2 (4-5 sentences): Map the candidate's most relevant experience to the job requirements. Reference specific projects, roles, tools, and outcomes from the resume. Use the same language the job posting uses. Be concrete — mention metrics, tools, and results.
- Paragraph 3 (3-4 sentences): Why the candidate would be a good fit, tying together skills and the role. Include a call to action. Mention the candidate's website if they have one. Thank the reader.

After paragraph 3:
- blank line
- "Sincerely"
- Candidate's full name
- Contact line: "email | phone | website" (extract from resume)

WRITING RULES:
- Sound like a real person, not a template. No corporate fluff.
- Keep the tone professional but approachable — confident without being arrogant.
- Total length: 250-400 words (one page when printed).
- Do NOT fabricate experience. Only reference what exists in the resume.
- Do NOT use phrases like "I am writing to express my interest" or "I believe I would be a great asset."
- Vary sentence length. Mix short punchy sentences with longer ones.
- Output ONLY the cover letter text. No markdown, no code fences, no commentary.

--- CANDIDATE'S RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{job_description}

Write the cover letter now:"""


@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    job_description = data.get("job_description", "").strip()
    model = data.get("model", DEFAULT_MODEL).strip()

    if not resume_text or not job_description:
        return {"error": "Resume and job description are required."}, 400

    prompt = build_cover_letter_prompt(resume_text, job_description)

    def stream_response():
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
