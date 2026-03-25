"""
test_reminders.py — Automated tests for the TABS Reminder System.
Run with: python test_reminders.py
No Ollama instance required (LLM parser is mocked for these tests).
"""

import os
import sys
import json
import datetime
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from reminders.reminder_store     import ReminderStore
from reminders.reminder_engine    import ReminderEngine
from reminders.reminder_responder import ReminderResponder


# ─── Mock helpers ─────────────────────────────────────────────────────────────

class MockLLM:
    """Stub LLM that returns a hard-coded JSON for parse tests."""
    def get_full_response(self, messages):
        return '{"title": "Workshop", "event_type": "event", "datetime_iso": "2026-03-20T10:00:00", "confidence": 0.9}'


spoken_messages = []

def mock_speak(text):
    spoken_messages.append(text)


# ─── Test: ReminderStore ──────────────────────────────────────────────────────

class TestReminderStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".json")
        self.store = ReminderStore(store_path=self.tmp)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_add_and_load(self):
        dt = datetime.datetime(2026, 3, 20, 10, 0)
        r  = self.store.add("Workshop", "event", dt)
        self.assertEqual(r["title"], "Workshop")
        self.assertEqual(r["event_type"], "event")
        self.assertIn(1440, r["alert_offsets"])  # 1 day
        self.assertIn(60,   r["alert_offsets"])  # 1 hour
        self.assertIn(15,   r["alert_offsets"])  # 15 min

        # Reload from disk
        store2 = ReminderStore(store_path=self.tmp)
        self.assertEqual(len(store2), 1)
        self.assertEqual(store2.get_all()[0]["title"], "Workshop")

    def test_delete(self):
        dt = datetime.datetime(2026, 3, 20, 10, 0)
        r  = self.store.add("Team standup", "meeting", dt)
        self.assertEqual(len(self.store), 1)
        deleted = self.store.delete_by_title("standup")
        self.assertIsNotNone(deleted)
        self.assertEqual(len(self.store), 0)

    def test_update(self):
        dt  = datetime.datetime(2026, 3, 20, 10, 0)
        r   = self.store.add("Meeting", "meeting", dt)
        new_dt = datetime.datetime(2026, 3, 20, 17, 0)
        self.store.update(r["id"], datetime=new_dt)
        updated = self.store.get_by_id(r["id"])
        self.assertIn("17:00", updated["datetime"])

    def test_get_for_date(self):
        today = datetime.datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
        self.store.add("Today's event", "event", today)
        results = self.store.get_for_date(today.date())
        self.assertEqual(len(results), 1)

    def test_get_upcoming(self):
        now  = datetime.datetime.now()
        dt1  = now + datetime.timedelta(days=2)
        dt2  = now + datetime.timedelta(days=10)  # outside window
        self.store.add("Near event",  "event", dt1)
        self.store.add("Far event",   "event", dt2)
        upcoming = self.store.get_upcoming(now, days=7)
        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming[0]["title"], "Near event")

    def test_mark_alert_fired(self):
        dt = datetime.datetime(2026, 3, 20, 10, 0)
        r  = self.store.add("Test", "event", dt)
        self.store.mark_alert_fired(r["id"], 60)
        updated = self.store.get_by_id(r["id"])
        self.assertIn("60", updated["fired_alerts"])

    def test_smart_defaults_birthday(self):
        dt = datetime.datetime(2026, 4, 1, 9, 0)
        r  = self.store.add("Mom's Birthday", "birthday", dt)
        self.assertEqual(r["alert_offsets"], [1440])

    def test_smart_defaults_task(self):
        dt = datetime.datetime(2026, 4, 1, 23, 59)
        r  = self.store.add("Submit assignment", "task", dt)
        self.assertIn(480,  r["alert_offsets"])
        self.assertIn(60,   r["alert_offsets"])


# ─── Test: ReminderResponder ──────────────────────────────────────────────────

class TestReminderResponder(unittest.TestCase):

    def setUp(self):
        self.tmp   = tempfile.mktemp(suffix=".json")
        self.store = ReminderStore(store_path=self.tmp)
        self.resp  = ReminderResponder(self.store)
        self.now   = datetime.datetime(2026, 3, 8, 16, 0, 0)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_confirm_added(self):
        dt = datetime.datetime(2026, 3, 20, 10, 0)
        r  = self.store.add("Workshop", "event", dt)
        msg = self.resp.confirm_added(r, self.now)
        self.assertIn("Workshop", msg)
        self.assertIn("Got it", msg)
        self.assertTrue(len(msg) > 0)

    def test_confirm_deleted(self):
        r  = {"title": "Stand-up", "id": "abc"}
        msg = self.resp.confirm_deleted(r)
        self.assertIn("Stand-up", msg)
        self.assertIn("removed", msg)

    def test_answer_today_empty(self):
        msg = self.resp.answer_today([], self.now)
        self.assertIn("no reminders", msg.lower())

    def test_answer_today_with_events(self):
        today_dt = self.now.replace(hour=11, minute=0, second=0, microsecond=0)
        reminders = [{"title": "Team Sync", "datetime": today_dt.isoformat()}]
        msg = self.resp.answer_today(reminders, self.now)
        self.assertIn("Team Sync", msg)
        self.assertIn("1", msg)  # count

    def test_answer_upcoming_empty(self):
        msg = self.resp.answer_upcoming([], self.now)
        self.assertIn("no upcoming", msg.lower())

    def test_list_all_empty(self):
        msg = self.resp.list_all([], self.now)
        self.assertIn("no reminders", msg.lower())

    def test_list_all_with_events(self):
        dt = datetime.datetime(2026, 3, 20, 10, 0)
        self.store.add("Workshop", "event", dt)
        msg = self.resp.list_all(self.store.get_all(), self.now)
        self.assertIn("Workshop", msg)


# ─── Test: ReminderEngine alert logic ────────────────────────────────────────

class TestReminderEngineAlerts(unittest.TestCase):

    def setUp(self):
        self.tmp        = tempfile.mktemp(suffix=".json")
        self.state_tmp  = tempfile.mktemp(suffix=".json")
        self.store      = ReminderStore(store_path=self.tmp)
        spoken_messages.clear()
        # Patch STATE_PATH for engine
        import reminders.reminder_engine as eng_mod
        self._orig_state = eng_mod.STATE_PATH
        eng_mod.STATE_PATH = self.state_tmp
        self.engine = ReminderEngine(self.store, mock_speak)

    def tearDown(self):
        import reminders.reminder_engine as eng_mod
        eng_mod.STATE_PATH = self._orig_state
        for p in (self.tmp, self.state_tmp):
            if os.path.exists(p):
                os.remove(p)

    def test_alert_fires_within_window(self):
        # Create a reminder that should have a 15-min alert trigger right now
        now    = datetime.datetime.now()
        event  = now + datetime.timedelta(minutes=15)
        r      = self.store.add("Stand-up", "meeting", event)

        spoken_messages.clear()
        self.engine.check_alerts_now()

        self.assertTrue(
            any("Stand-up" in m for m in spoken_messages),
            f"Expected spoken alert for Stand-up. Got: {spoken_messages}"
        )
        # Confirm it was marked fired
        updated = self.store.get_by_id(r["id"])
        self.assertIn("15", updated["fired_alerts"])

    def test_alert_does_not_fire_twice(self):
        now   = datetime.datetime.now()
        event = now + datetime.timedelta(minutes=15)
        r     = self.store.add("Demo", "meeting", event)

        spoken_messages.clear()
        self.engine.check_alerts_now()
        count_after_first = len(spoken_messages)

        self.engine.check_alerts_now()  # second pass — should NOT re-fire
        count_after_second = len(spoken_messages)

        self.assertEqual(count_after_first, count_after_second,
                         "Alert should not fire a second time.")

    def test_no_alert_for_far_future(self):
        now   = datetime.datetime.now()
        event = now + datetime.timedelta(hours=48)  # 48 hours (avoids the 24h default alert)
        self.store.add("Future Event", "event", event)

        spoken_messages.clear()
        self.engine.check_alerts_now()

        # No alert should fire (no offset matches right now)
        self.assertFalse(
            any("Future Event" in m for m in spoken_messages),
            f"Should not alert for a far-future event. Got: {spoken_messages}"
        )


# ─── Test: Morning digest generation ─────────────────────────────────────────

class TestDigestGeneration(unittest.TestCase):

    def setUp(self):
        self.tmp   = tempfile.mktemp(suffix=".json")
        self.store = ReminderStore(store_path=self.tmp)
        import reminders.reminder_engine as eng_mod
        self._orig_state = eng_mod.STATE_PATH
        eng_mod.STATE_PATH = tempfile.mktemp(suffix=".json")
        self.engine = ReminderEngine(self.store, mock_speak)

    def tearDown(self):
        import reminders.reminder_engine as eng_mod
        eng_mod.STATE_PATH = self._orig_state
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_digest_contains_sections(self):
        now      = datetime.datetime.now()
        today    = now.replace(hour=11, minute=0, second=0, microsecond=0)
        tomorrow = (now + datetime.timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        future   = (now + datetime.timedelta(days=4)).replace(hour=14, minute=0, second=0, microsecond=0)

        self.store.add("Team Sync",     "meeting", today)
        self.store.add("Mom's Birthday","birthday", tomorrow)
        self.store.add("Workshop",      "event",   future)

        digest = self.engine.generate_digest_text(now)

        self.assertIn("Good Morning", digest)
        self.assertIn("Team Sync",      digest)
        self.assertIn("Mom's Birthday", digest)
        self.assertIn("Workshop",       digest)

    def test_empty_digest_when_no_reminders(self):
        digest = self.engine.generate_digest_text()
        self.assertEqual(digest, "")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  TABS Reminder System — Automated Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
