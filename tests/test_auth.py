import pytest

from core.auth import (
    generate_raw_key,
    extract_key_prefix,
    hash_api_key,
    verify_api_key,
    KEY_PREFIX,
    KEY_PREFIX_LENGTH,
)


class TestKeyGeneration:
    def test_generate_raw_key_format(self):
        key = generate_raw_key()
        assert key.startswith(KEY_PREFIX)
        assert len(key) > len(KEY_PREFIX)

    def test_generate_raw_key_uniqueness(self):
        keys = {generate_raw_key() for _ in range(100)}
        assert len(keys) == 100

    def test_extract_key_prefix(self):
        key = generate_raw_key()
        prefix = extract_key_prefix(key)
        assert prefix == key[:KEY_PREFIX_LENGTH]
        assert prefix.startswith(KEY_PREFIX)
        assert len(prefix) == KEY_PREFIX_LENGTH

    def test_extract_key_prefix_invalid_format(self):
        with pytest.raises(ValueError):
            extract_key_prefix("invalid_key")

    def test_extract_key_prefix_wrong_prefix(self):
        with pytest.raises(ValueError):
            extract_key_prefix("sk_invalid_abcdefghijklmnop")


class TestKeyHashing:
    def test_hash_api_key_returns_bcrypt_hash(self):
        key = generate_raw_key()
        hashed = hash_api_key(key)
        assert hashed.startswith("$2b$12$")
        assert len(hashed) == 60

    def test_hash_api_key_deterministic_for_same_input_with_same_salt(self):
        key = generate_raw_key()
        hashed1 = hash_api_key(key)
        hashed2 = hash_api_key(key)
        assert hashed1 != hashed2

    def test_verify_api_key_valid(self):
        key = generate_raw_key()
        hashed = hash_api_key(key)
        assert verify_api_key(key, hashed) is True

    def test_verify_api_key_invalid(self):
        key1 = generate_raw_key()
        key2 = generate_raw_key()
        hashed = hash_api_key(key1)
        assert verify_api_key(key2, hashed) is False

    def test_verify_api_key_wrong_format(self):
        hashed = hash_api_key(generate_raw_key())
        assert verify_api_key("not_a_valid_key", hashed) is False

    def test_verify_api_key_constant_time(self):
        import time
        key = generate_raw_key()
        hashed = hash_api_key(key)

        valid_times = []
        invalid_times = []

        for _ in range(100):
            start = time.perf_counter()
            verify_api_key(key, hashed)
            valid_times.append(time.perf_counter() - start)

            wrong_key = generate_raw_key()
            start = time.perf_counter()
            verify_api_key(wrong_key, hashed)
            invalid_times.append(time.perf_counter() - start)

        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)

        diff_ratio = abs(avg_valid - avg_invalid) / max(avg_valid, avg_invalid)
        assert diff_ratio < 0.5