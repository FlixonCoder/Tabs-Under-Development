"""
Reminder Responder — Natural language response builder for TABS reminder actions.
Generates human-friendly confirmation and summary messages for all reminder operations.
"""

import datetime
from typing import List, Dict, Optional


def _fmt_time(dt: datetime.datetime) -> str:
    """Format a datetime as a readable time string, e.g. '4:00 PM'."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _fmt_date(dt: datetime.datetime, now: datetime.datetime) -> str:
    """Return a human-friendly date relative to today."""
    today    = now.date()
    tomorrow = today + datetime.timedelta(days=1)
    d        = dt.date()

    if d == today:
        return "today"
    if d == tomorrow:
        return "tomorrow"
    if (d - today).days < 7:
        return f"on {dt.strftime('%A')}"  # "on Wednesday"
    return f"on {dt.strftime('%B')} {int(dt.strftime('%d'))}"


def _fmt_datetime(dt: datetime.datetime, now: datetime.datetime) -> str:
    """Full date-time description: 'tomorrow at 4 PM'."""
    date_part = _fmt_date(dt, now)
    time_part = _fmt_time(dt)
    return f"{date_part} at {time_part}"


def _offsets_to_words(offsets: List[int]) -> str:
    """Convert a list of minute offsets to a human-readable schedule sentence."""
    parts = []
    for m in sorted(offsets):
        if m == 0:
            parts.append("at the time of the event")
        elif m < 60:
            parts.append(f"{m} minutes before")
        elif m == 60:
            parts.append("1 hour before")
        elif m < 1440:
            hours = m // 60
            parts.append(f"{hours} hours before")
        elif m == 1440:
            parts.append("1 day before")
        else:
            days = m // 1440
            parts.append(f"{days} days before")

    if not parts:
        return "no alerts scheduled"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


class ReminderResponder:
    """Generates conversational responses for all reminder-related actions."""

    def __init__(self, store=None):
        """store: ReminderStore instance (optional, used for richer responses)."""
        self._store = store

    # ──────────────────────────────────────────────────────────────────────────
    # Confirmation messages
    # ──────────────────────────────────────────────────────────────────────────

    def confirm_added(
        self,
        reminder: dict,
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """
        Returns a natural confirmation when a reminder is created.
        e.g. "Got it. I've added a reminder for your Meeting tomorrow at 4 PM.
              I'll remind you 1 day before, 1 hour before, and 15 minutes before."
        """
        if now is None:
            now = datetime.datetime.now()

        try:
            dt = datetime.datetime.fromisoformat(reminder["datetime"])
        except Exception:
            dt = now

        title      = reminder.get("title", "event")
        offsets    = reminder.get("alert_offsets", [])
        schedule   = _offsets_to_words(offsets)
        when       = _fmt_datetime(dt, now)

        return (
            f"Got it. I have added a reminder for your {title} {when}. "
            f"I will remind you {schedule}."
        )

    def confirm_deleted(self, reminder: dict) -> str:
        """'Done. I have removed the reminder for your Workshop.'"""
        title = reminder.get("title", "event")
        return f"Done. I have removed the reminder for your {title}."

    def confirm_not_found(self, keyword: str) -> str:
        return f"I couldn't find any reminder matching '{keyword}'."

    def confirm_updated(
        self,
        reminder: dict,
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """'Done. I have updated your Meeting to Tuesday at 5 PM.'"""
        if now is None:
            now = datetime.datetime.now()
        try:
            dt = datetime.datetime.fromisoformat(reminder["datetime"])
        except Exception:
            dt = now
        title = reminder.get("title", "event")
        when  = _fmt_datetime(dt, now)
        return f"Done. I have updated your {title} to {when}."

    # ──────────────────────────────────────────────────────────────────────────
    # Query responses
    # ──────────────────────────────────────────────────────────────────────────

    def answer_today(
        self,
        reminders: List[Dict],
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """Respond to 'what reminders do I have today?'"""
        if now is None:
            now = datetime.datetime.now()
        if not reminders:
            return "You have no reminders for today, Sir."
        count = len(reminders)
        noun  = "reminder" if count == 1 else "reminders"
        items = " ".join(
            f"{r['title']} at {_fmt_time(datetime.datetime.fromisoformat(r['datetime']))}."
            for r in reminders
        )
        return f"You have {count} {noun} today, Sir. {items}"

    def answer_tomorrow(
        self,
        reminders: List[Dict],
        now: Optional[datetime.datetime] = None,
    ) -> str:
        if now is None:
            now = datetime.datetime.now()
        if not reminders:
            return "You have nothing scheduled for tomorrow, Sir."
        items = " ".join(
            f"{r['title']} at {_fmt_time(datetime.datetime.fromisoformat(r['datetime']))}."
            for r in reminders
        )
        return f"Tomorrow you have: {items}"

    def answer_upcoming(
        self,
        reminders: List[Dict],
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """Respond to 'what's coming up?' / 'next 7 days'."""
        if now is None:
            now = datetime.datetime.now()
        if not reminders:
            return "You have no upcoming reminders in the next 7 days, Sir."
        lines = []
        for r in reminders:
            try:
                dt   = datetime.datetime.fromisoformat(r["datetime"])
                when = _fmt_datetime(dt, now)
                lines.append(f"{r['title']} {when}")
            except Exception:
                lines.append(r["title"])
        items = ". ".join(lines)
        return f"Upcoming events, Sir: {items}."

    def answer_next_event(
        self,
        reminders: List[Dict],
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """Respond to 'when is my next event?'"""
        if now is None:
            now = datetime.datetime.now()
        future = [
            r for r in reminders
            if datetime.datetime.fromisoformat(r["datetime"]) >= now
        ]
        if not future:
            return "You have no upcoming events, Sir."
        nxt  = future[0]
        try:
            dt   = datetime.datetime.fromisoformat(nxt["datetime"])
            when = _fmt_datetime(dt, now)
        except Exception:
            when = "soon"
        return f"Your next event is {nxt['title']}, {when}."

    def list_all(
        self,
        reminders: List[Dict],
        now: Optional[datetime.datetime] = None,
    ) -> str:
        """Return a concise spoken list of all reminders."""
        if now is None:
            now = datetime.datetime.now()
        if not reminders:
            return "You have no reminders saved, Sir."
        lines = []
        for r in reminders:
            try:
                dt   = datetime.datetime.fromisoformat(r["datetime"])
                when = _fmt_datetime(dt, now)
                lines.append(f"{r['title']} {when}")
            except Exception:
                lines.append(r["title"])
        items = ". ".join(lines)
        count = len(reminders)
        noun  = "reminder" if count == 1 else "reminders"
        return f"You have {count} {noun}, Sir: {items}."
