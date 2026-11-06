"""An orphaned alarm must be stoppable after Streamlit drops its registry."""
import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

from core.st_utils import handoff_alarm


class AlarmStopTests(unittest.TestCase):
    def test_stop_orphan_after_reload(self):
        played = threading.Event()
        backend = types.SimpleNamespace(
            SND_MEMORY=1, SND_NODEFAULT=2,
            PlaySound=lambda *args: played.set())
        with patch.dict(sys.modules, winsound=backend), patch.object(sys, 'platform', 'win32'):
            alarm = handoff_alarm.HandoffAlarm(interval=60)
            self.assertTrue(played.wait(2))
            try:
                handoff_alarm._active_alarm = None
                importlib.reload(handoff_alarm)
                self.assertGreaterEqual(handoff_alarm.stop_active_alarm(), 1)
                self.assertFalse(alarm._thread.is_alive())
            finally:
                alarm._stop.set()
                alarm._thread.join(2)


if __name__ == '__main__':
    unittest.main()
