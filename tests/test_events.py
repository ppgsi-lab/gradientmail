"""Callbacks combined into one, and the one that logs."""

import logging

import pytest

from gradientmail.events import Event, LeaseReleased, PollFailed, combined, log_events


def describe_combined():
    def it_hands_every_event_to_each_callback_in_order():
        seen: list[tuple[str, Event]] = []
        notify = combined(lambda e: seen.append(("a", e)), lambda e: seen.append(("b", e)))

        notify(LeaseReleased(node="node-0", lease="learner"))

        assert seen == [
            ("a", LeaseReleased("node-0", "learner")),
            ("b", LeaseReleased("node-0", "learner")),
        ]

    def when_a_callback_raises():
        def it_logs_the_failure_and_still_calls_the_rest(caplog: pytest.LogCaptureFixture):
            def failing(event: Event) -> None:
                raise RuntimeError("sink down")

            seen: list[Event] = []
            notify = combined(failing, seen.append)

            notify(LeaseReleased("node-0", "learner"))

            assert seen == [LeaseReleased("node-0", "learner")]
            [failure] = [r for r in caplog.records if r.levelno == logging.ERROR]
            assert failure.getMessage() == "node-0: a callback failed on LeaseReleased"


def describe_log_events():
    def it_logs_a_failure_with_the_traceback_it_carries(caplog: pytest.LogCaptureFixture):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            error = exc

        log_events(PollFailed("node-0", error))

        [record] = caplog.records
        assert (record.levelno, record.getMessage()) == (
            logging.ERROR,
            "node-0: poll failed; retrying",
        )
        assert record.exc_info is not None
        assert record.exc_info[1] is error
