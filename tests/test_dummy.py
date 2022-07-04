import pytest

def capital_case(x):
    return x.capitalize()

def upper_case(x):
    return x.upper()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'

def test_upper_case():
    assert upper_case('semaphore') == 'SEMAPHORE'
    
