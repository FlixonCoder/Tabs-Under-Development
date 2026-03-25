"""
Reminder Parser — Intent detection and datetime extraction for TABS reminders.

Primary path : Regex-based datetime parsing (fast, reliable, no LLM call)
Secondary    : LLM JSON extraction for ambiguous cases (e.g. "next Thursday")
Fallback     : If LLM JSON also fails, regex result still used if time found

This two-stage approach means a reminder is ALWAYS stored when the user
expresses intent, even if Qwen2.5:3b returns conversational text instead of JSON.
"""

import re
import json
import datetime
from typing import Optional, Tuple

# ─── Intent keyword sets ──────────────────────────────────────────────────────

_ADD_KEYWORDS = {
    # explicit reminder words
    "remind me", "remind", "reminder", "remindme",
    "set a reminder", "set up a reminder", "set reminder",
    "add a reminder", "add reminder", "create a reminder",
    "schedule", "set an alarm", "add an event",
    # event type words (strong signals when combined with time)
    "birthday", "meeting", "appointment", "event", "task",
    "deadline", "call",
    # imperative phrases
    "don't let me forget", "dont let me forget",
    "don't forget", "dont forget",
    "note that", "make a note",
    "at", "today", "tomorrow",   # will be validated with time presence
}

_QUERY_KEYWORDS = {
    "what reminders", "my reminders", "any reminders",
    "upcoming reminders", "list reminders", "show reminders",
    "next event", "what events", "today's reminders",
    "tomorrow's reminder", "next 7 days", "upcoming events",
    "what's on", "whats on", "what do i have",
    "any events", "my schedule", "any upcoming",
    "how many reminders", "do i have any",
}

# Strong negative check: if the query keyword set matches, don't treat as add
_STRONG_QUERY_MARKERS = {
    "what", "do i have", "any reminders", "show", "list",
    "upcoming", "my schedule",
}

_DELETE_KEYWORDS = {
    "delete", "remove", "cancel", "clear", "forget about",
}

_EDIT_KEYWORDS = {
    "change", "update", "edit", "reschedule", "move", "postpone",
    "shift", "modify",
}

_LIST_KEYWORDS = {
    "list", "show", "display", "give me", "tell me my",
}

# ─── Regex patterns for time/date extraction ─────────────────────────────────

# Matches: "4:50 PM", "4 PM", "10:30 am", "16:50", "1650", "4 50 pm" (spoken)
_TIME_PATTERN = re.compile(
    r"""\b
    (?:
        (\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?   # HH:MM [am/pm]        groups 0,1,2
        |
        (\d{1,2})\s+(\d{2})\s+(am|pm)        # H MM am/pm (spoken)  groups 3,4,5
        |
        (\d{1,2})\s+(am|pm)                   # H am/pm               groups 6,7
        |
        (\d{4})\s*(am|pm)?                    # HHMM                  groups 8,9
    )
    \b""",
    re.VERBOSE | re.IGNORECASE,
)

# Relative day words
_TODAY_WORDS    = re.compile(r"\btoday\b", re.IGNORECASE)
_TOMORROW_WORDS = re.compile(r"\btomorrow\b", re.IGNORECASE)
_WEEKDAY_WORDS  = re.compile(
    r"\b(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
# Absolute dates: "March 15", "15 March", "march 15 2026"
_ABS_DATE_PATTERN = re.compile(
    r"""\b
    (?:
        (january|february|march|april|may|june|july|august|
         september|october|november|december)\s+(\d{1,2})(?:\s+(\d{4}))?
        |
        (\d{1,2})\s+(january|february|march|april|may|june|july|august|
                      september|october|november|december)(?:\s+(\d{4}))?
    )
    \b""",
    re.VERBOSE | re.IGNORECASE,
)

_MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}
_WEEKDAY_MAP = {
    "monday":0,"tuesday":1,"wednesday":2,"thursday":3,
    "friday":4,"saturday":5,"sunday":6,
}

# Event type detection
_BIRTHDAY_RE = re.compile(r"\bbirthday\b", re.IGNORECASE)
_MEETING_RE  = re.compile(r"\b(meeting|standup|stand-up|sync)\b", re.IGNORECASE)
_TASK_RE     = re.compile(r"\b(deadline|submit|task|assignment|homework)\b", re.IGNORECASE)
_CALL_RE     = re.compile(r"\b(call|phone)\b", re.IGNORECASE)

# ─── LLM prompt template (used only as secondary enrichment) ─────────────────

_PARSE_PROMPT = """\
Extract reminder details from the text below. Reply ONLY with a JSON object.

Reference time: {now_str}
Text: "{text}"

Rules:
- title: short event name (2-4 words max)
- event_type: one of [birthday, meeting, task, event, call]
- datetime_iso: ISO 8601 (YYYY-MM-DDTHH:MM:SS) using reference time for relative words
- confidence: 0.0-1.0

JSON only, no other text: {{"title":"...","event_type":"...","datetime_iso":"...","confidence":0.9}}"""


class ReminderParser:
    """
    Detects reminder intents and extracts structured datetime data.
    Uses regex as primary parser (always reliable) with LLM as secondary enrichment.
    """

    def __init__(self, llm):
        self._llm = llm

    # ──────────────────────────────────────────────────────────────────────────
    # Intent detection
    # ──────────────────────────────────────────────────────────────────────────

    def is_reminder_intent(self, text: str) -> bool:
        """Return True if text is a reminder-creation request."""
        lower = text.lower()

        # Explicit "remind" family — always a strong signal
        if re.search(r"\bremind\b", lower):
            return True

        # Check for explicit "set/add/create reminder/alarm" phrases
        if re.search(r"\b(set|add|create)\s+(up\s+)?(a\s+)?(reminder|alarm|event)\b", lower):
            return True

        # Event keywords + a time expression = intent
        has_event_word = bool(re.search(
            r"\b(birthday|meeting|appointment|task|deadline|call|schedule)\b", lower
        ))
        has_time = bool(_TIME_PATTERN.search(text))
        has_relative = bool(
            _TODAY_WORDS.search(lower) or
            _TOMORROW_WORDS.search(lower) or
            _WEEKDAY_WORDS.search(lower)
        )
        if has_event_word and (has_time or has_relative):
            return True

        return False

    def is_reminder_query(self, text: str) -> bool:
        """Return True if text is asking about existing reminders."""
        lower = text.lower()
        return any(kw in lower for kw in _QUERY_KEYWORDS)

    def is_reminder_management(self, text: str) -> Optional[Tuple[str, str]]:
        """Detect delete/edit/list management commands."""
        lower = text.lower()
        if any(kw in lower for kw in _DELETE_KEYWORDS):
            return ("delete", text)
        if any(kw in lower for kw in _EDIT_KEYWORDS):
            # Don't mistake "change to 5 PM" for a management command
            # if there's also a "remind" — that's a new reminder
            if not re.search(r"\bremind\b", lower):
                return ("edit", text)
        if any(kw in lower for kw in _LIST_KEYWORDS):
            if any(kw in lower for kw in {"reminder", "reminders", "event", "events", "schedule"}):
                return ("list", text)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Parsing — two-stage
    # ──────────────────────────────────────────────────────────────────────────

    def parse_reminder(
        self,
        text: str,
        now: Optional[datetime.datetime] = None,
    ) -> Optional[dict]:
        """
        Extract structured reminder data from free text.

        Stage 1: Fast regex extraction (title guess + time + type).
        Stage 2: LLM JSON for enrichment / ambiguous cases.
        Returns None only if absolutely no datetime can be found.
        """
        if now is None:
            now = datetime.datetime.now()

        # Stage 1 — regex
        regex_result = self._regex_parse(text, now)

        # Stage 2 — try LLM for enrichment (better title + type)
        llm_result = self._llm_parse(text, now)

        # Merge: prefer LLM fields when confident, fall back to regex
        if llm_result and llm_result.get("confidence", 0) >= 0.5:
            # Use LLM title/type, but use regex datetime if LLM datetime looks wrong
            result = llm_result.copy()
            if regex_result and llm_result["event_datetime"].date() < now.date():
                # LLM gave a past date — trust the regex date instead
                result["event_datetime"] = regex_result["event_datetime"]
            return result

        # LLM failed or low confidence — use regex result
        return regex_result

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Regex parser
    # ──────────────────────────────────────────────────────────────────────────

    def _regex_parse(
        self, text: str, now: datetime.datetime
    ) -> Optional[dict]:
        """
        Best-effort regex extraction. Always returns a result if a time is found,
        even if title/type guesses are rough.
        """
        # 1. Extract time
        parsed_time = self._extract_time(text)
        if parsed_time is None:
            return None  # Cannot create a reminder without a time

        hour, minute = parsed_time

        # 2. Extract date
        event_date = self._extract_date(text, now)

        # 3. Build full datetime
        event_dt = datetime.datetime.combine(
            event_date,
            datetime.time(hour=hour, minute=minute, second=0)
        )
        # If time is in the past today, interpret as tomorrow
        if event_dt <= now:
            event_dt += datetime.timedelta(days=1)

        # 4. Guess event type
        etype = self._guess_type(text)

        # 5. Guess title
        title = self._guess_title(text, etype)

        return {
            "title":          title,
            "event_type":     etype,
            "event_datetime": event_dt,
            "confidence":     0.7,
        }

    def _extract_time(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract (hour24, minute) from text. Returns None if no time found."""
        m = _TIME_PATTERN.search(text)
        if not m:
            return None

        g = m.groups()
        # Groups: (hh,mm,ampm1) | (h,mi,ampm2) | (h,ampm3) | (hhmm,ampm4)
        #          0  1  2          3  4  5          6  7          8    9

        if g[0] is not None:        # HH:MM [am/pm]
            h, mi = int(g[0]), int(g[1])
            ampm  = (g[2] or "").lower()
        elif g[3] is not None:      # H MM am/pm  (spoken: "4 50 pm")
            h, mi = int(g[3]), int(g[4])
            ampm  = (g[5] or "").lower()
        elif g[6] is not None:      # H am/pm
            h, mi = int(g[6]), 0
            ampm  = (g[7] or "").lower()
        else:                       # HHMM (e.g. 1650)
            raw   = int(g[8])
            h, mi = raw // 100, raw % 100
            ampm  = (g[9] or "").lower()

        if ampm == "pm" and h < 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0

        if 0 <= h <= 23 and 0 <= mi <= 59:
            return (h, mi)
        return None

    def _extract_date(self, text: str, now: datetime.datetime) -> datetime.date:
        """
        Extract event date. Priority: absolute date > tomorrow > weekday > today.
        Defaults to today if nothing found.
        """
        lower = text.lower()

        # Absolute date: "March 20", "20 March 2026"
        m = _ABS_DATE_PATTERN.search(lower)
        if m:
            g = m.groups()
            if g[0]:  # "Month DD [YYYY]"
                month  = _MONTH_MAP[g[0].lower()]
                day    = int(g[1])
                year   = int(g[2]) if g[2] else now.year
            else:      # "DD Month [YYYY]"
                day    = int(g[3])
                month  = _MONTH_MAP[g[4].lower()]
                year   = int(g[5]) if g[5] else now.year
            try:
                d = datetime.date(year, month, day)
                if d < now.date():
                    d = d.replace(year=d.year + 1)
                return d
            except ValueError:
                pass

        # Tomorrow
        if _TOMORROW_WORDS.search(lower):
            return now.date() + datetime.timedelta(days=1)

        # Named weekday (next Monday, etc.)
        wd_m = _WEEKDAY_WORDS.search(lower)
        if wd_m:
            target_wd  = _WEEKDAY_MAP[wd_m.group(1).lower()]
            current_wd = now.weekday()
            days_ahead = (target_wd - current_wd) % 7
            if days_ahead == 0:
                days_ahead = 7  # "next Monday" always means next week
            return now.date() + datetime.timedelta(days=days_ahead)

        # Today (explicit or default)
        return now.date()

    def _guess_type(self, text: str) -> str:
        if _BIRTHDAY_RE.search(text):  return "birthday"
        if _MEETING_RE.search(text):   return "meeting"
        if _TASK_RE.search(text):      return "task"
        if _CALL_RE.search(text):      return "event"  # treat calls as events
        return "event"

    def _guess_title(self, text: str, etype: str) -> str:
        """Extract a short title by stripping command words."""
        lower = text.lower()
        # Strip leading command phrases
        for phrase in [
            "remind me to", "remind me about", "remind me of", "remind me",
            "set up a reminder for", "set a reminder for", "set reminder for",
            "add a reminder for", "add reminder for", "create a reminder for",
            "schedule a", "schedule an", "set an alarm for",
            "don't let me forget", "dont let me forget",
            "don't forget", "dont forget",
            "note that", "make a note of",
            "i have a", "i have an", "i've got a", "there is a", "there's a",
        ]:
            if lower.startswith(phrase):
                lower = lower[len(phrase):].strip()
                break

        # Strip trailing time/date phrases
        lower = re.sub(
            r"\b(today|tomorrow|tonight|this\s+\w+|next\s+\w+|"
            r"on\s+\w+|at\s+[\d:apmAPM ]+|in\s+\d+\s+\w+)\b.*$",
            "", lower
        ).strip()

        # Capitalise
        title = " ".join(w.capitalize() for w in lower.split()) if lower else etype.capitalize()
        # Trim to 40 chars
        return title[:40] or etype.capitalize()

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: LLM enrichment
    # ──────────────────────────────────────────────────────────────────────────

    def _llm_parse(
        self, text: str, now: datetime.datetime
    ) -> Optional[dict]:
        """
        Ask the LLM for structured JSON. Used as enrichment, not primary path.
        Returns None on failure instead of propagating exceptions.
        """
        now_str  = now.strftime("%Y-%m-%d %H:%M (%A, %Z)")
        prompt   = _PARSE_PROMPT.format(now_str=now_str, text=text)
        messages = [{"role": "user", "content": prompt}]
        try:
            raw    = self._llm.get_full_response(messages)
            parsed = self._extract_json(raw)
            if not parsed:
                return None

            title  = parsed.get("title", "").strip()
            etype  = parsed.get("event_type", "event").lower().strip()
            dt_iso = parsed.get("datetime_iso", "").strip()
            conf   = float(parsed.get("confidence", 0.5))

            if not title or not dt_iso:
                return None
            event_dt = self._parse_iso(dt_iso, now)
            if event_dt is None:
                return None
            if etype not in {"birthday", "meeting", "task", "event", "call"}:
                etype = "event"

            return {
                "title":          title,
                "event_type":     etype,
                "event_datetime": event_dt,
                "confidence":     conf,
            }
        except Exception as e:
            print(f"⚠️ ReminderParser LLM error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Edit helpers
    # ──────────────────────────────────────────────────────────────────────────

    def extract_title_keyword(self, text: str) -> str:
        """Strip command verbs to get a title keyword for delete/edit lookups."""
        lower = text.lower()
        for phrase in [
            "delete my", "remove my", "cancel my", "delete the", "remove the",
            "cancel the", "forget about", "change my", "update my", "edit my",
            "reschedule my", "move my", "postpone my",
            "delete", "remove", "cancel", "change", "update", "edit",
            "reschedule", "move", "postpone", "reminder", "reminders",
        ]:
            lower = lower.replace(phrase, " ")
        lower = re.sub(r"\breminder\b", "", lower)
        return " ".join(lower.split()).strip()

    def extract_new_time(
        self,
        text: str,
        now: Optional[datetime.datetime] = None,
    ) -> Optional[datetime.datetime]:
        """
        For edit commands, extract the new desired datetime.
        Uses regex first (reliable), then LLM as fallback.
        """
        if now is None:
            now = datetime.datetime.now()

        # Regex path
        parsed = self._extract_time(text)
        if parsed:
            h, mi    = parsed
            new_date = self._extract_date(text, now)
            dt       = now.replace(
                year=new_date.year, month=new_date.month, day=new_date.day,
                hour=h, minute=mi, second=0, microsecond=0,
            )
            if dt <= now:
                dt += datetime.timedelta(days=1)
            return dt

        # LLM path
        prompt = (
            f"Current datetime: {now.strftime('%Y-%m-%d %H:%M')}.\n"
            f"Extract ONLY the new date/time as ISO 8601 (YYYY-MM-DDTHH:MM:SS). "
            f"No other text.\nText: \"{text}\""
        )
        try:
            raw = self._llm.get_full_response([{"role": "user", "content": prompt}]).strip()
            return self._parse_iso(raw, now)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Greedy: grab the LAST {...} block (LLMs sometimes prefix with explanation)
        matches = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
        for m in reversed(matches):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _parse_iso(dt_str: str, reference: datetime.datetime) -> Optional[datetime.datetime]:
        dt_str = re.sub(r"[+-]\d{2}:\d{2}$", "", dt_str.strip()).strip()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
        ):
            try:
                return datetime.datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        return None
