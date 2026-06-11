"""Email banner resolution and dynamic (LLM-edited) banner generation."""

import os
from pathlib import Path

from pipeline.common.use_predictions.llm import resolve_claude_model

# For direct Anthropic API calls
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

# For image generation
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None


def _resolve_banner_path():
    project_root = Path(__file__).resolve().parents[3]
    configured = os.getenv("FOOTY_TIPPER_EMAIL_BANNER")

    candidates = []
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        candidates.append(configured_path)
    candidates.append(project_root / "images" / "email-banner.png")

    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    return None


def _build_banner_edit_instruction(copy, anthropic_client, news_context=None, news_hit=None):
    """Ask Claude for a fun, topical scenario for the two banner characters this week."""
    subject = copy.get("subject", "")
    opening = copy.get("opening", "")[:300]
    # news_hit is the primary source — it's already the most interesting story distilled
    if news_hit:
        inspiration = f"This week's big story (PRIMARY inspiration for the banner):\n{news_hit}"
    elif news_context:
        inspiration = f"NRL news this week:\n{news_context}"
    else:
        inspiration = f"Email subject: {subject}\nEmail opening: {opening}"
    response = anthropic_client.messages.create(
        model=resolve_claude_model(),
        system="You write short, vivid image editing instructions for a fun weekly sports email banner.",
        messages=[{"role": "user", "content": (
            f"A weekly NRL tipping email banner features two cartoon characters: Reg Reagan (a bloke in a shirt that says 'Bring Back the Biff' wearing green and gold Australian rugby league footy shorts) and a dingo. "
            f"Come up with a funny or energetic scenario for this week's banner inspired by the content below. "
            f"Put Reg and the dingo in a situation that directly references the story or themes — they can be doing anything: celebrating, arguing, cowering, riding something, holding a sign, dressed up, etc. "
            f"Be creative and specific.\n\n"
            f"{inspiration}\n\n"
            "Return 2-3 sentences describing the scene. Be visual and specific. No preamble."
        )}],
        max_tokens=150,
        temperature=1.0,
    )
    topical = response.content[0].text.strip()
    return (
        f"Reimagine this image as a wide landscape email banner. "
        f"CRITICAL FRAMING RULE: Every character and every element must be completely within the frame — do not crop any part of any character at any edge. Use a wide establishing shot with clear margins on all sides. "
        f"The 'Reg's Footy Tips' logo badge must be FULLY VISIBLE and CENTRED in the image — do not crop or push it to an edge. "
        f"Maintain the original composition: one character on the far left with room to breathe, the logo badge prominently in the centre, the other character on the far right with room to breathe. "
        f"The two characters are Reg Reagan (a bloke whose shirt reads 'Bring Back the Biff' wearing green and gold Australian rugby league footy shorts) and a dingo — both must be shown in full from head to toe, fully inside the canvas. "
        f"Maintain the same overall visual style, colour palette, and brand aesthetic as the original: bright blue background with circuit-board pattern. "
        f"Scene: {topical} "
        f"Fun, punchy sports editorial illustration style."
    )



def _generate_dynamic_banner(copy, anthropic_api_key, openai_api_key, news_context=None, news_hit=None):
    """Edit the existing email banner with topical elements via Claude + gpt-image-1."""
    if not anthropic_api_key or not openai_api_key:
        return None
    if Anthropic is None or OpenAIClient is None:
        print("Dynamic banner skipped: Anthropic or OpenAI SDK unavailable.")
        return None
    try:
        import base64
        import io
        from PIL import Image
    except ImportError:
        print("Dynamic banner skipped: Pillow not installed.")
        return None

    try:
        project_root = Path(__file__).resolve().parents[3]
        banner_path = project_root / "images" / "email-banner.png"
        if not banner_path.exists():
            print("Dynamic banner skipped: base banner not found.")
            return None

        anthropic_client = Anthropic(api_key=anthropic_api_key)
        edit_instruction = _build_banner_edit_instruction(copy, anthropic_client, news_context=news_context, news_hit=news_hit)
        print(f"Banner edit: {edit_instruction[:120]}...")

        img = Image.open(banner_path).convert("RGBA")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        openai_client = OpenAIClient(api_key=openai_api_key)
        response = openai_client.images.edit(
            model="gpt-image-1.5",
            image=("email-banner.png", img_bytes, "image/png"),
            prompt=edit_instruction,
            size="1536x1024",
        )
        image_data = base64.b64decode(response.data[0].b64_json)
        out_path = project_root / "images" / "email-banner-generated.png"
        out_path.write_bytes(image_data)
        print(f"Dynamic banner saved: {out_path.name}")
        return str(out_path)
    except Exception as exc:
        import traceback
        print(f"Dynamic banner generation failed: {exc}")
        traceback.print_exc()
        return None
