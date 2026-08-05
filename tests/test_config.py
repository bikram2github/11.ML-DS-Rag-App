import os

def test_env_variables_exist():
    required_vars = [
        "GROQ_API_KEY",
        "PG_DB",
        "PG_USER",
        "PG_PASSWORD",
        "PG_HOST",
        "PG_PORT"
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    assert not missing, f"Missing env vars: {missing}"
