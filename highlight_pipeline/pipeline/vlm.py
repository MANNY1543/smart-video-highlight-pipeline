"""VLM scene understanding via the hosted Gemini API (google-genai SDK).

One multimodal request per keyframe (clean image only, never the video) returns a structured
{caption, exciting, reason} object enforced by a Pydantic/JSON schema. The result is validated with
VLMResult; exactly one retry (corrective prompt, same image); then a clearly marked fallback.
The API key is read only from the GEMINI_API_KEY environment variable and is never logged.
"""
import logging
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

log = logging.getLogger(__name__)

PROMPT = (
    "You are an expert sports-video event analyst. Carefully inspect the supplied keyframe and identify the "
    "sport and the most specific event that is genuinely supported by visible evidence. Do not merely describe "
    "a player standing on the field if a clear sporting action or event is visible. Look for the ball and its "
    "position, player movement, body posture, interaction between players, goal/wicket/net/field area, official "
    "signals, celebrations, and readable scoreboard or broadcast graphics. Distinguish the event itself from the "
    "surrounding scene. For cricket, consider delivery, batting shot, boundary attempt, apparent four/six only "
    "when supported by visible evidence or readable graphics, running between wickets, catch attempt, fielding, "
    "run-out attempt, wicket/appeal, umpire signal, celebration, and general play. For football, consider pass, "
    "shot, goal, save, tackle, foul, penalty, corner, free kick, and celebration. For basketball, consider "
    "dribble, pass, shot attempt, made basket only when supported, rebound, block, steal, dunk, and celebration. "
    "For other sports, identify the most specific clearly visible event or use a concise general event label. "
    "You are analyzing a single image, not a full video: do not infer an outcome that occurs outside the frame, "
    "and do not claim a goal, wicket, boundary, successful shot, score change, or other completed event unless "
    "the image itself provides evidence. You may use readable broadcast graphics as evidence, but distinguish "
    "what the graphic states from what the image visually shows. Never invent player names, teams, scores, "
    "locations, match results, or unseen actions. If the event cannot be identified confidently from this frame, "
    "set event_type to general_play or uncertain and explicitly state the limitation in event_description. Set "
    "exciting=true only when visible action, a notable event, a celebration, a clear competitive moment, or "
    "reliable visible score/context supports highlight value; do not mark every frame exciting merely because it "
    "belongs to a close match. Return only the required structured response.")

PROMPT_RETRY = (
    "Re-analyze this image using only visible evidence. Return exactly the required structured fields: caption, "
    "sport, event_type, event_description, players_and_actions, exciting and reason. Use event_type "
    "general_play or uncertain if the specific event is not clearly visible, and say so in event_description. "
    "exciting must be true or false. Do not invent names, scores or outcomes that are not visible.")

USER_TEXT = "Analyze this image."

_DESC = {
    "caption": "One concise, image-grounded sentence describing the visible scene or action. Visible evidence only.",
    "sport": "The sport or activity visible in the frame, lowercase (e.g. cricket, football, basketball, tennis), "
             "or 'unknown' if it cannot be told.",
    "event_type": "The most specific event supported by visible evidence, as a short lowercase snake_case label "
                  "(e.g. batting_shot, bowling_delivery, boundary_four, six, wicket, run_out, catch, goal, "
                  "penalty, save, dunk, shot_attempt, serve, rally, celebration). Use 'general_play' or "
                  "'uncertain' when no specific event is clearly visible. Never claim a completed outcome "
                  "(goal, wicket, four, six) without evidence in the image.",
    "event_description": "One sentence on what specifically appears to happen. If event_type is general_play or "
                         "uncertain, state the limitation (what cannot be determined from this single frame).",
    "players_and_actions": "The visible player actions and interactions (e.g. 'batter follows through; fielder "
                           "runs towards the ball'), without guessing identities or names not readable in the image.",
    "exciting": "Boolean. true only when visible action, a notable event, a celebration, a clear competitive "
                "moment or reliable visible score/context supports highlight value; false otherwise.",
    "reason": "Short justification of the exciting decision using only visible evidence.",
}


def _one_sentence(v):
    v = v.strip()
    return re.split(r"(?<=[.!?])\s+", v)[0] if v else ""


class VLMResult(BaseModel):
    """Exactly seven fields; extra keys and string booleans are rejected."""
    model_config = ConfigDict(extra="forbid")
    caption: str = Field(description=_DESC["caption"])
    sport: str = Field(description=_DESC["sport"])
    event_type: str = Field(description=_DESC["event_type"])
    event_description: str = Field(description=_DESC["event_description"])
    players_and_actions: str = Field(description=_DESC["players_and_actions"])
    exciting: StrictBool = Field(description=_DESC["exciting"])
    reason: str = Field(description=_DESC["reason"])

    @field_validator("caption")
    @classmethod
    def _real_caption(cls, v):
        v = v.strip()
        if len(re.findall(r"[A-Za-z]{2,}", v)) < 2:
            raise ValueError("caption too short to be a real description")
        return _one_sentence(v)

    @field_validator("sport", "event_type")
    @classmethod
    def _label(cls, v):                      # normalise to lowercase snake_case, must be non-empty
        v = re.sub(r"[^a-z0-9]+", "_", v.strip().lower()).strip("_")
        if not v:
            raise ValueError("label must not be empty")
        return v

    @field_validator("event_description")
    @classmethod
    def _real_event_description(cls, v):
        v = v.strip()
        if len(re.findall(r"[A-Za-z]{2,}", v)) < 3:
            raise ValueError("event_description too short to be a real description")
        return _one_sentence(v)

    @field_validator("players_and_actions", "reason")
    @classmethod
    def _text(cls, v):
        return v.strip()


class GeminiSchema(BaseModel):
    """Response schema sent to Gemini (no extra='forbid'/validators); re-validated with VLMResult."""
    caption: str = Field(description=_DESC["caption"])
    sport: str = Field(description=_DESC["sport"])
    event_type: str = Field(description=_DESC["event_type"])
    event_description: str = Field(description=_DESC["event_description"])
    players_and_actions: str = Field(description=_DESC["players_and_actions"])
    exciting: bool = Field(description=_DESC["exciting"])
    reason: str = Field(description=_DESC["reason"])


_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class VLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        self._last_call = 0.0
        self.stats = {"ok": 0, "retry_ok": 0, "fallback": 0}
        self.last_raw = []
        self.enabled = cfg.vlm_model.lower() != "none"
        if self.enabled:
            try:
                self._load()
            except Exception as e:
                log.error("Could not start Gemini client (%s). Using fallback for ALL shots.", self._redact(e))
                self.enabled = False

    def _redact(self, e):
        s = str(e)
        key = os.environ.get("GEMINI_API_KEY", "")
        return s.replace(key, "***") if key else s

    def _load(self):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        from google import genai
        from google.genai import types
        self._types = types
        # timeout is in milliseconds; the single retry is handled by analyse(), not the SDK
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(self.cfg.vlm_timeout_sec * 1000)))
        log.info("Gemini VLM ready (model=%s, remote inference, no local GPU needed)", self.cfg.vlm_model)

    def _throttle(self):
        wait = self.cfg.vlm_min_interval_sec - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)                      # stay under free-tier requests-per-minute limits
        self._last_call = time.time()

    def _generate(self, image_path, prompt):
        """One multimodal request: the clean keyframe + prompt -> raw structured JSON text."""
        t = self._types
        p = Path(image_path)
        image = t.Part.from_bytes(data=p.read_bytes(), mime_type=_MIME.get(p.suffix.lower(), "image/jpeg"))
        self._throttle()
        resp = self.client.models.generate_content(
            model=self.cfg.vlm_model,
            contents=[image, prompt],
            config=t.GenerateContentConfig(
                system_instruction=PROMPT,
                response_mime_type="application/json",
                response_schema=GeminiSchema,
                temperature=0.2,
                max_output_tokens=self.cfg.vlm_max_output_tokens))
        if not resp.text:
            raise ValueError("empty response (blocked or no candidates)")
        return resp.text

    def analyse(self, image_path):
        """Returns (VLMResult, status): 'ok', 'retry_ok' or 'fallback'. Never raises."""
        self.last_raw = []
        if self.enabled:
            for attempt, prompt in enumerate((USER_TEXT, PROMPT_RETRY)):    # attempt 1 = the single retry
                try:
                    raw = self._generate(image_path, prompt)
                    self.last_raw.append(raw)
                    log.debug("VLM raw output (attempt %d): %r", attempt + 1, raw)
                    res = VLMResult.model_validate_json(raw)
                    status = "ok" if attempt == 0 else "retry_ok"
                    self.stats[status] += 1
                    return res, status
                except (ValueError, ValidationError) as e:
                    log.warning("VLM output rejected (attempt %d): %s", attempt + 1, str(e).splitlines()[0])
                except Exception as e:                  # rate limit, timeout, network, API error
                    log.warning("Gemini call failed (attempt %d): %s: %s", attempt + 1, type(e).__name__,
                                self._redact(e)[:200])
                if attempt == 0:
                    time.sleep(self.cfg.vlm_retry_delay_sec)
        self.stats["fallback"] += 1
        return (VLMResult(caption="[fallback] No VLM description available.", sport="unknown", event_type="uncertain",
                          event_description="[fallback] No event analysis available: VLM unavailable or invalid output.",
                          players_and_actions="", exciting=False,
                          reason="[fallback] VLM unavailable or output invalid after one retry."), "fallback")
