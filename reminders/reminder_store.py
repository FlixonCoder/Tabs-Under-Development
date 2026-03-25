"""
Reminder Store — Persistent JSON-backed storage for TABS reminders.
All reminders are saved to data/reminders.json and loaded on startup.
"""

import os
import json
import uuid
import datetime
from typing import List, Dict, Optional


STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reminders.json")

# Default alert offsets in minutes before the event
DEFAULT_OFFSETS = {
    "birthday": [1440],          # 1 day before
    "meeting":  [1440, 60, 15],  # 1 day, 1 hour, 15 min
    "event":    [1440, 60, 15],  # same as meeting
    "task":     [480, 60],       # morning digest equivalent (~8 AM == ~480 min from midnight) + 1 hour
}


class ReminderStore:
    """
    Manages reminder persistence. Each reminder is a dict:
    {
        "id":           str (UUID),
        "title":        str,
        "event_type":   str  (birthday | meeting | task | event),
        "datetime":     str  (ISO 8601, e.g. "2026-03-20T10:00:00"),
        "alert_offsets":list[int]  (minutes before event),
        "fired_alerts": list[str]  (keys like "1440", "60", "15" already fired),
        "created_at":   str  (ISO 8601),
    }
    """

    def __init__(self, store_path: str = STORE_PATH):
        self._path = store_path
        self._reminders: List[Dict] = []
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._load()

    # ──────────────────────────────────────────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        event_type: str,
        event_datetime: datetime.datetime,
        alert_offsets: Optional[List[int]] = None,
    ) -> Dict:
        """
        Add a new reminder. Returns the created reminder dict.
        If alert_offsets is None, smart defaults are applied based on event_type.
        """
        etype = event_type.lower() if event_type else "event"
        if alert_offsets is None:
            alert_offsets = DEFAULT_OFFSETS.get(etype, DEFAULT_OFFSETS["event"])

        reminder = {
            "id":            str(uuid.uuid4()),
            "title":         title,
            "event_type":    etype,
            "datetime":      event_datetime.isoformat(),
            "alert_offsets": alert_offsets,
            "fired_alerts":  [],
            "created_at":    datetime.datetime.now().isoformat(),
        }
        self._reminders.append(reminder)
        self._save()
        return reminder

    def get_all(self) -> List[Dict]:
        """Return all reminders (sorted by datetime ascending)."""
        return sorted(self._reminders, key=lambda r: r["datetime"])

    def get_by_id(self, reminder_id: str) -> Optional[Dict]:
        """Return a specific reminder by ID, or None."""
        for r in self._reminders:
            if r["id"] == reminder_id:
                return r
        return None

    def find_by_title(self, keyword: str) -> List[Dict]:
        """Return reminders whose title contains keyword (case-insensitive)."""
        kw = keyword.lower()
        return [r for r in self._reminders if kw in r["title"].lower()]

    def delete(self, reminder_id: str) -> bool:
        """Delete a reminder by ID. Returns True if found and deleted."""
        before = len(self._reminders)
        self._reminders = [r for r in self._reminders if r["id"] != reminder_id]
        if len(self._reminders) < before:
            self._save()
            return True
        return False

    def delete_by_title(self, keyword: str) -> Optional[Dict]:
        """Delete the first reminder matching keyword. Returns deleted reminder or None."""
        matches = self.find_by_title(keyword)
        if not matches:
            return None
        target = matches[0]
        self.delete(target["id"])
        return target

    def update(self, reminder_id: str, **kwargs) -> Optional[Dict]:
        """
        Update fields on an existing reminder.
        Accepted kwargs: title, event_type, datetime (str ISO or datetime obj),
                         alert_offsets, fired_alerts.
        Returns updated reminder or None if not found.
        """
        reminder = self.get_by_id(reminder_id)
        if not reminder:
            return None

        if "title" in kwargs:
            reminder["title"] = kwargs["title"]
        if "event_type" in kwargs:
            reminder["event_type"] = kwargs["event_type"]
        if "datetime" in kwargs:
            dt = kwargs["datetime"]
            reminder["datetime"] = dt.isoformat() if isinstance(dt, datetime.datetime) else dt
        if "alert_offsets" in kwargs:
            reminder["alert_offsets"] = kwargs["alert_offsets"]
        if "fired_alerts" in kwargs:
            reminder["fired_alerts"] = kwargs["fired_alerts"]

        self._save()
        return reminder

    def mark_alert_fired(self, reminder_id: str, offset_minutes: int):
        """Record that a specific alert offset has been triggered."""
        reminder = self.get_by_id(reminder_id)
        if reminder:
            key = str(offset_minutes)
            if key not in reminder["fired_alerts"]:
                reminder["fired_alerts"].append(key)
                self._save()

    def mark_fired(self, reminder_id: str, offset_minutes: int):
        """Alias for mark_alert_fired."""
        self.mark_alert_fired(reminder_id, offset_minutes)


    # ──────────────────────────────────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────────────────────────────────

    def get_for_date(self, date: datetime.date) -> List[Dict]:
        """Return reminders whose event date matches the given date."""
        target = date.isoformat()  # "YYYY-MM-DD"
        return [r for r in self._reminders if r["datetime"].startswith(target)]

    def get_upcoming(self, from_dt: datetime.datetime, days: int = 7) -> List[Dict]:
        """Return reminders within the next `days` days from from_dt."""
        cutoff = from_dt + datetime.timedelta(days=days)
        result = []
        for r in self._reminders:
            try:
                rdt = datetime.datetime.fromisoformat(r["datetime"])
                if from_dt <= rdt <= cutoff:
                    result.append(r)
            except ValueError:
                pass
        return sorted(result, key=lambda r: r["datetime"])

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._reminders, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ ReminderStore save error: {e}")

    def _load(self):
        if not os.path.exists(self._path):
            self._reminders = []
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._reminders = data
            else:
                self._reminders = []
        except Exception as e:
            print(f"⚠️ ReminderStore load error: {e}")
            self._reminders = []

    def __len__(self):
        return len(self._reminders)


if __name__ == "__main__":
    store = ReminderStore()
    import datetime as dt
    r = store.add("Team standup", "meeting", dt.datetime(2026, 3, 20, 10, 0))
    print("Added:", r["title"], "→", r["datetime"])
    print("All reminders:", len(store.get_all()))
