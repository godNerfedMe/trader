import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.leverage import interest
from freqtrade.util import FtPrecise


@pytest.fixture
def borrowed_amount():
    return FtPrecise(60.0)


@pytest.fixture
def exchange_params():
    return {
        'binance': {
            'interest_rate': 0.00025,
            'ten_mins_expected': 0.000625,
            'five_hours_expected': 0.003125,
            'twentyfive_hours_expected': 0.015625,
        },
        'kraken': {
            'interest_rate': 0.00025,
            'ten_mins_expected': 0.03,
            'five_hours_expected': 0.045,
            'twentyfive_hours_expected': 0.12,
        },
    }


def test_interest(exchange, borrowed_amount, exchange_params):
    params = exchange_params[exchange]

    def assert_interest(hours, expected):
        result = interest(exchange_name=exchange, borrowed=borrowed_amount,
                          rate=FtPrecise(params['interest_rate']), hours=hours)
        assert pytest.approx(float(result)) == expected

    assert_interest(1 / 6, params['ten_mins_expected'])
    assert_interest(5.0, params['five_hours_expected'])
    assert_interest(25.0, params['twentyfive_hours_expected'])


def test_interest_exception():
    with pytest.raises(OperationalException, match=r"Leverage not available on .* with freqtrade"):
        interest(exchange_name='bitmex', borrowed=FtPrecise(60.0),
                 rate=FtPrecise(0.0005), hours=1 / 6)
