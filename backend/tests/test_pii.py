"""Tests for PII masking. Run from backend/:  pip install pytest && python -m pytest"""

import pytest

from app.guardrails.pii import mask_pii, privacy_notice


@pytest.mark.parametrize(
    "text, label",
    [
        ("My Aadhaar is 2345 6789 0123", "AADHAAR"),
        ("aadhaar 234567890123 please update", "AADHAAR"),
        ("PAN: ABCDE1234F", "PAN"),
        ("my pan is abcde1234f", "PAN"),
        ("card 4111 1111 1111 1111 was blocked", "CARD_NUMBER"),
        ("card 4111-1111-1111-1111", "CARD_NUMBER"),
        ("call me on +91 98765 43210", "PHONE"),
        ("my number 9876543210", "PHONE"),
        ("account number 001234567890123", "ACCOUNT_NUMBER"),
        ("email me at seema.test@example.com", "EMAIL"),
        ("pay to name@okbank", "UPI_ID"),
        ("my OTP is 482913", "SECRET"),
        ("mpin: 1234", "SECRET"),
        ("CVV 123", "SECRET"),
    ],
)
def test_masks_sensitive_values(text, label):
    result = mask_pii(text)
    assert f"[{label}]" in result.text
    assert label in result.found


def test_value_is_removed():
    result = mask_pii("My Aadhaar is 2345 6789 0123 and PAN ABCDE1234F")
    assert "2345" not in result.text and "ABCDE1234F" not in result.text
    assert result.found == ["AADHAAR", "PAN"]


@pytest.mark.parametrize(
    "text",
    [
        "What documents do I need for KYC?",
        "Will the bank officer ask for my OTP during the video call?",
        "Is a PIN code needed for the address proof?",
        "Minimum balance is Rs 5,000 and closure costs Rs 500 within 12 months",
        "Can I keep Rs 1,00,000 in a small account?",
        "Call customer care on 1800-000-0000",
        "Re-KYC every 2 years, 8 years or 10 years",
        "Is a 4111 1111 1111 1112 card valid?",  # fails the Luhn check
    ],
)
def test_leaves_normal_questions_alone(text):
    result = mask_pii(text)
    assert result.text == text
    assert not result.masked


def test_privacy_notice():
    assert privacy_notice([]) is None
    notice = privacy_notice(["AADHAAR", "PAN", "AADHAAR"])
    assert "Aadhaar number and PAN were removed" in notice
    assert "PAN was removed" in privacy_notice(["PAN"])
