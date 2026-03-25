"""
Reminder Engine — Real-time background clock for TABS reminder alerts and morning digest.
Ticks every 1 second and fires alerts the moment the scheduled time is reached.
"""

import os
import sys
import json
import time
import threading
import datetime
from typing import Callable, Optional

from reminders.reminder_store import ReminderStore

# Ringtone sitting next to this file in reminders/
RINGTONE_PATH = os.path.join(os.path.dirname(__file__), "the_good_the_bad_t.mp3")

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reminder_state.json")

# Morning digest window: 07:00 – 09:00 local time
DIGEST_HOUR_START = 7
DIGEST_HOUR_END   = 9

# How tightly the clock matches an alert trigger (seconds either side)
# With a 1-second tick this just absorbs any tiny clock drift
ALERT_WINDOW_SECONDS = 5

# Clock tick rate — 1 second for real-time feel
CLOCK_TICK_SECONDS = 1

# Show a live status line in the console every N ticks (0 = disable)
STATUS_EVERY_TICKS = 60   # once per minute


class ReminderEngine:
    """
    Real-time background clock that:
    1. Ticks every 1 second — fires alerts the moment the clock matches.
    2. Delivers a morning digest once per day between 07:00–09:00.
    3. Prints a live status line periodically so you can see it's running.
    """

    def __init__(
        self,
        store: ReminderStore,
        speak_fn: Callable[[str], None],
        print_fn: Optional[Callable[[str], None]] = None,
    ):
        self._store     = store
        self._speak     = speak_fn
        self._print     = print_fn or print
        self._stop_evt  = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True, name="ReminderClock")
        self._state     = self._load_state()
        # Dedup set: (reminder_id, offset_key, minute_str) already fired this run
        self._fired_this_session: set = set()

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        """Start the real-time background clock."""
        self._thread.start()
        self._print("\u23f0 Reminder Clock started — tracking reminders in real-time.")

    def stop(self):
        """Signal the scheduler thread to stop and wait for it."""
        self._stop_evt.set()
        self._thread.join(timeout=5)
        self._print("🛑 Reminder Engine stopped.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers (also callable directly for synchronous checks)
    # ──────────────────────────────────────────────────────────────────────────

    def check_morning_digest_now(self):
        """
        Call once at startup to deliver the digest if we are in the digest
        window and haven't sent it today yet.
        """
        self._check_morning_digest(datetime.datetime.now())

    def check_alerts_now(self):
        """
        Check for any due alerts right now (synchronous). Useful for testing.
        """
        self._check_alerts(datetime.datetime.now())

    def generate_digest_text(self, now: Optional[datetime.datetime] = None) -> str:
        """
        Build the morning digest string without speaking it.
        """
        if now is None:
            now = datetime.datetime.now()
        return self._build_digest(now)

    # ──────────────────────────────────────────────────────────────────────────
    # Background loop
    # ──────────────────────────────────────────────────────────────────────────

    def _run(self):
        """Real-time 1-second clock loop."""
        tick = 0
        while not self._stop_evt.is_set():
            try:
                now = datetime.datetime.now()

                # — Morning digest (checked every tick, gated by date) —
                self._check_morning_digest(now)

                # — Alert sweep —
                self._check_alerts(now)

                # — Periodic status line —
                if STATUS_EVERY_TICKS and tick % STATUS_EVERY_TICKS == 0:
                    pending = self._count_pending(now)
                    self._print(
                        f"\r⏰ [{now.strftime('%H:%M:%S')}] "
                        f"Reminder clock running — {pending} pending alert(s)",
                        end="  ",
                        flush=True,
                    )

                tick += 1
            except Exception as e:
                self._print(f"\n⚠️ ReminderClock error: {e}")

            self._stop_evt.wait(CLOCK_TICK_SECONDS)

    # ──────────────────────────────────────────────────────────────────────────
    # Morning digest
    # ──────────────────────────────────────────────────────────────────────────

    def _check_morning_digest(self, now: datetime.datetime):
        """Send digest once per day if inside the digest window."""
        today_str = now.date().isoformat()
        if self._state.get("last_digest_date") == today_str:
            return  # Already sent today

        if DIGEST_HOUR_START <= now.hour < DIGEST_HOUR_END:
            digest = self._build_digest(now)
            if digest:
                self._print(f"\n☀️  [Morning Digest]\n{digest}\n")
                self._speak(digest)
                self._state["last_digest_date"] = today_str
                self._save_state()

    def _build_digest(self, now: datetime.datetime) -> str:
        """Build the morning digest text."""
        today    = now.date()
        tomorrow = today + datetime.timedelta(days=1)

        today_reminders    = self._store.get_for_date(today)
        tomorrow_reminders = self._store.get_for_date(tomorrow)
        upcoming           = self._store.get_upcoming(
            now + datetime.timedelta(days=2), days=5
        )  # days 2–7 from now

        if not today_reminders and not tomorrow_reminders and not upcoming:
            return ""

        lines = ["Good Morning!"]

        if today_reminders:
            lines.append("Today's reminders:")
            for r in today_reminders:
                lines.append(self._format_short(r))

        if tomorrow_reminders:
            lines.append("Tomorrow:")
            for r in tomorrow_reminders:
                lines.append(self._format_short(r))

        if upcoming:
            lines.append("Upcoming this week:")
            for r in upcoming:
                lines.append(self._format_upcoming(r))

        return " ".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Alert checking
    # ──────────────────────────────────────────────────────────────────────────

    def _check_alerts(self, now: datetime.datetime):
        """Check all reminders and fire any due alerts."""
        for reminder in self._store.get_all():
            try:
                event_dt = datetime.datetime.fromisoformat(reminder["datetime"])
            except (ValueError, KeyError):
                continue

            for offset in reminder.get("alert_offsets", []):
                key = str(offset)

                # Skip if already persisted as fired
                if key in reminder.get("fired_alerts", []):
                    continue

                alert_dt = event_dt - datetime.timedelta(minutes=offset)

                # Fire if the current time has reached or passed the alert time,
                # and this specific alert offset hasn't been fired yet.
                # (We don't use a strict time window anymore, because if the CPU hangs
                # for 6 seconds, we'd permanently miss the alert. This way, it just plays a few seconds late.)
                if now >= alert_dt:
                    # Trigger the alert
                    self._trigger_alert(reminder, offset, event_dt)
                    
                    # Immediately mark as fired in persistent storage
                    # so we don't fire it again on the next loop 1 second later
                    self._store.mark_fired(reminder["id"], offset)
                    
                    # Also update the local dict so we don't fire other duplicate offsets
                    # (like if 0 and 15 both became due during a long CPU sleep)
                    reminder.setdefault("fired_alerts", []).append(key)

    def _play_ringtone(self):
        """
        Play the ringtone asynchronously so TTS can follow immediately.
        Uses winsound on Windows (built-in, no install). Falls back silently.
        """
        if not os.path.exists(RINGTONE_PATH):
            return
        try:
            if sys.platform == "win32":
                import winsound
                # SND_FILENAME | SND_ASYNC — plays in background, returns instantly
                winsound.PlaySound(RINGTONE_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Non-Windows: try sounddevice + soundfile (WAV only, may fail for MP3)
                import soundfile as sf
                import sounddevice as sd
                data, sr = sf.read(RINGTONE_PATH)
                sd.play(data, sr)
        except Exception as e:
            self._print(f"⚠️  Ringtone play error: {e}")

    def _trigger_alert(
        self,
        reminder: dict,
        minutes_before: int,
        event_dt: datetime.datetime,
    ):
        """
        Full alert sequence:
          1. Play the ringtone (async — heard immediately)
          2. Print a visible \033 chat banner in the console
          3. Speak the message via TTS
          4. Persist the alert as fired
        """
        title  = reminder["title"]
        etype  = reminder.get("event_type", "event").capitalize()
        time_s = event_dt.strftime("%I:%M %p").lstrip("0")
        date_s = event_dt.strftime("%A, %B %d")

        # ── Build the spoken / displayed message ──
        if minutes_before == 0:
            msg = (
                f"Sir, your {etype} '{title}' is happening right now! "
                f"It was scheduled for {time_s} today."
            )
        elif minutes_before < 60:
            msg = (
                f"Sir, heads up! Your {etype} '{title}' starts in just "
                f"{minutes_before} minutes, at {time_s}."
            )
        elif minutes_before == 60:
            msg = (
                f"Sir, your {etype} '{title}' is coming up in 1 hour, at {time_s}."
            )
        elif minutes_before < 1440:
            hours    = minutes_before // 60
            mins_rem = minutes_before % 60
            time_part = f"{hours} hour{'s' if hours != 1 else ''}"
            if mins_rem:
                time_part += f" and {mins_rem} minutes"
            msg = (
                f"Sir, reminder: your {etype} '{title}' is in {time_part}, at {time_s}."
            )
        else:
            days = minutes_before // 1440
            day_word = "tomorrow" if days == 1 else f"in {days} days"
            msg = (
                f"Sir, just a heads-up. Your {etype} '{title}' is {day_word}, "
                f"on {date_s} at {time_s}."
            )

        try:
            # Step 1: Play ringtone (async — starts immediately)
            self._play_ringtone()

            # Step 2: Print a visible chat banner in the console
            banner_width = 58
            self._print("\n")
            self._print("┌" + "─" * banner_width + "┐")
            self._print(f"│  🔔 TABS REMINDER                                       │")
            self._print("├" + "─" * banner_width + "┤")
            self._print(f"│  Event  : {title[:44]:<44}  │")
            self._print(f"│  Type   : {etype:<44}  │")
            self._print(f"│  Time   : {time_s:<44}  │")
            self._print(f"│  Alert  : {self._offset_label(minutes_before):<44}  │")
            self._print("└" + "─" * banner_width + "┘")
            self._print(f"\n🤖 TABS: {msg}\n")

            # Step 3: Speak via TTS
            try:
                self._speak(msg)
            except Exception as e:
                self._print(f"⚠️ TTS crash caught in alert thread: {e}")

            # Step 4: Mark alert as fired in the store
            self._store.mark_alert_fired(reminder["id"], minutes_before)
            
        except Exception as big_e:
            self._print(f"⚠️ FATAL in _trigger_alert: {big_e}")

    @staticmethod
    def _offset_label(minutes_before: int) -> str:
        """Human-readable label for an alert offset."""
        if minutes_before == 0:
            return "Now"
        elif minutes_before < 60:
            return f"{minutes_before} minutes before"
        elif minutes_before == 60:
            return "1 hour before"
        elif minutes_before < 1440:
            h = minutes_before // 60
            return f"{h} hour{'s' if h != 1 else ''} before"
        else:
            d = minutes_before // 1440
            return f"{d} day{'s' if d != 1 else ''} before"


    def _count_pending(self, now: datetime.datetime) -> int:
        """Count unfired alert offsets that are still in the future."""
        count = 0
        for reminder in self._store.get_all():
            try:
                event_dt = datetime.datetime.fromisoformat(reminder["datetime"])
            except (ValueError, KeyError):
                continue
            fired = reminder.get("fired_alerts", [])
            for offset in reminder.get("alert_offsets", []):
                if str(offset) not in fired:
                    alert_dt = event_dt - datetime.timedelta(minutes=offset)
                    if alert_dt > now:
                        count += 1
        return count

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._state, f)
        except Exception as e:
            self._print(f"⚠️ ReminderEngine state save error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Formatting helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_short(reminder: dict) -> str:
        try:
            dt   = datetime.datetime.fromisoformat(reminder["datetime"])
            time = dt.strftime("%I:%M %p").lstrip("0")
            return f"  {reminder['title']} at {time}."
        except Exception:
            return f"  {reminder['title']}."

    @staticmethod
    def _format_upcoming(reminder: dict) -> str:
        try:
            dt   = datetime.datetime.fromisoformat(reminder["datetime"])
            date = dt.strftime("%B %d")
            time = dt.strftime("%I:%M %p").lstrip("0")
            return f"  {reminder['title']} on {date} at {time}."
        except Exception:
            return f"  {reminder['title']}."
