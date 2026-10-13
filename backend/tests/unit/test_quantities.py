import pytest

from groundedai.core.quantities import Quantity, figure_present, read_quantities


def _one(text: str) -> Quantity:
    quantities = read_quantities(text).quantities
    assert len(quantities) == 1, quantities
    return quantities[0]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("The deposit is €500.", "A deposit of 500 euros is required."),
        ("The fee is EUR 45.", "It costs €45."),
        ("Members may book up to two sessions a week.", "Members may book up to 2 sessions."),
        ("Open from 08:00.", "Open from 8 am."),
        ("Closes at 18:00.", "Closes at 6 pm."),
        ("Loans last three weeks.", "Loans last 21 days."),
        ("Bills arrive every two months.", "Bills arrive every 2 months."),
        ("The limit is twelve people.", "The limit is 12 visitors."),
    ],
)
def test_the_same_value_written_differently_is_the_same_figure(first: str, second: str) -> None:
    assert _one(first).same_value(_one(second))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("E-books are lent for 14 days.", "E-books are lent for three weeks."),
        ("Replace it every 3 months.", "Replace it twice a year."),
        ("The pool opens at 06:30.", "The pool opens at 7:00."),
        ("Items can be renewed three times.", "Items can be renewed twice."),
    ],
)
def test_different_values_of_the_same_kind_differ(first: str, second: str) -> None:
    a, b = _one(first), _one(second)
    assert a.comparable(b)
    assert not a.same_value(b)


def test_working_days_are_not_calendar_days() -> None:
    a, b = _one("Paid within 10 working days."), _one("Submit within 30 days.")
    assert not a.comparable(b)
    assert _one("3 business days").unit == "working_day"


def test_identifiers_are_not_figures() -> None:
    reading = read_quantities("The SX-5 shows E07; the enclosure is IP65 and SEV1 pages on call.")
    assert reading.quantities == []
    assert reading.identifiers == ["SX-5", "E07", "IP65", "SEV1"]


def test_decimal_money_is_not_a_time_but_a_closing_time_is() -> None:
    assert _one("Cars are reimbursed at €0.30 per kilometre.").kind == "money"
    closing = _one("Access on weekdays until 20:00.")
    assert (closing.kind, closing.value) == ("time_of_day", 1200.0)


def test_dates_years_and_month_names() -> None:
    assert _one("From 1 March 2026 the fee rises.").value == 20260301.0
    assert _one("Passenger FAQ (2024 edition)").kind == "year"
    assert read_quantities("Employees may book hotels in May without approval.").quantities == []


def test_ranges_share_their_unit() -> None:
    low, high = read_quantities("Delivery takes 3 to 5 working days.").quantities
    assert (low.written, high.written, low.unit) == (3.0, 5.0, "working_day")
    volts = read_quantities("Input range 150 to 850 V.").quantities
    assert [q.value for q in volts] == [150.0, 850.0]


def test_adjectives_are_not_frequencies() -> None:
    assert read_quantities("The daily meal allowance applies.").quantities == []
    assert _one("Water the beds daily.").interval


def test_rates_record_their_period() -> None:
    fee = _one("Parking costs €2.50 per hour.")
    assert (fee.kind, fee.value, fee.per) == ("money", 2.5, "hour")


def test_figure_present_is_lenient_about_units_for_bare_numbers() -> None:
    passage = read_quantities("Claims must be made within 28 days.").quantities
    assert figure_present(_one("[28]"), passage)
    assert not figure_present(_one("[14]"), passage)
    assert figure_present(_one("within 10 days"), read_quantities("10 working days").quantities)
    assert not figure_present(_one("€28"), passage)
