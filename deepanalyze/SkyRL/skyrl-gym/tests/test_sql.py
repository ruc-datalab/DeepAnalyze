"""
uv run --isolated --extra dev pytest tests/test_sql.py
"""

import skyrl_gym
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig
from skyrl_gym.envs.sql.utils import verify_format_and_extract

# Mock data for testing
MOCK_DB_RESULTS = {
    "SELECT name FROM employees WHERE id = 1;": [("John Doe",)],
    "SELECT name FROM employees WHERE id = 2;": [("Jane Smith",)],
    "GOLDEN_SQL;": [("John Doe",)],
}


@pytest.fixture
def mock_db_file():
    with patch("os.path.exists") as mock_exists:
        # Make the mock return True for any database file path
        mock_exists.return_value = True
        yield mock_exists


@pytest.fixture
def mock_sqlite_connection():
    with patch("sqlite3.connect") as mock_connect:
        # Create a mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup the mock cursor's execute and fetchall methods
        def mock_execute(sql, *args, **kwargs):
            if sql in MOCK_DB_RESULTS:
                mock_cursor.fetchall.return_value = MOCK_DB_RESULTS[sql]
            else:
                mock_cursor.fetchall.return_value = []

        mock_cursor.execute = mock_execute
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        yield mock_connect


@pytest.mark.parametrize(
    "step_1_output, step_2_output, ground_truth, expected",
    [
        # Test case 1: matching sql and solution
        (
            "<think>a</think>\n\n\
         <sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 1;</solution>",
            "SELECT name FROM employees WHERE id = 1;",
            1.0,
        ),
        # Test case 2: identical sql but different solution
        (
            "<think>a</think>\n\n\
         <sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 1;</solution>",
            "GOLDEN_SQL;",
            1.0,
        ),
        # Test case 3: solution is wrong
        (
            "<think>a</think>\n\n\
         <sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 2;</solution>",
            "SELECT name FROM employees WHERE id = 1;",
            0.0,
        ),
        # Test case 4: includes invalid sql msg
        (
            "<think>a</think>",  # no sql tag here
            "<think>b</think>\n\n\
         <solution>SELECT name FROM employees WHERE id = 1;</solution>",
            "SELECT name FROM employees WHERE id = 1;",
            1.0,
        ),
    ],
    ids=[
        "correct_sql_and_solution",
        "golden_sql_same_result",
        "wrong_solution",
        "invalid_sql_msg",
    ],
)
def test_compute_score(
    mock_db_file,
    mock_sqlite_connection,
    step_1_output,
    step_2_output,
    ground_truth,
    expected,
):
    extras = {
        "reward_spec": {
            "method": "rule",
            "ground_truth": "SELECT name FROM employees WHERE id = 1;",
        },
        "max_turns": 3,
        "db_id": "test_db",
        "data": "spider",
    }
    env = skyrl_gym.make(
        "text2sql",
        env_config=DictConfig({"db_path": "/home/ray/default/sql_data"}),
        extras=extras,
    )
    # Skip init() since it's not used in this test

    reminder_text = "<reminder>You have 2 turns left to complete the task.</reminder>"
    invalid_sql_message = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."

    output = env.step(step_1_output)
    obs1 = output["observations"]
    reward = output["reward"]

    # intermediate step reward is 0
    assert reward == 0.0
    # check reminder message
    assert reminder_text in obs1[0]["content"]
    if "<sql>" not in step_1_output:
        assert invalid_sql_message in obs1[0]["content"]
    output = env.step(step_2_output)
    reward = output["reward"]

    # Only assert reward when done is True
    if output["done"]:
        assert reward == expected
    else:
        assert reward == 0.0


@pytest.mark.parametrize(
    "output, parse_ground_truth",
    [
        # Test case 1: Valid SQL query
        (
            "<sql>SELECT name FROM employees WHERE id = 1;</sql>",
            "SELECT name FROM employees WHERE id = 1;",
        ),
        # Test case 3: Invalid formatting
        ("<sql>SELECT name FROM employees", None),
    ],
)
def test_tool_parsing(mock_db_file, mock_sqlite_connection, output, parse_ground_truth):
    extras = {
        "reward_spec": {
            "method": "rule",
            "ground_truth": "SELECT name FROM employees WHERE id = 1;",
        },
        "max_turns": 2,
        "db_id": "test_db",
        "data": "spider",
    }
    env = skyrl_gym.make(
        "text2sql",
        env_config=DictConfig({"db_path": "/home/ray/default/sql_data"}),
        extras=extras,
    )
    # Skip init() since it's not used in this test

    # Step once and get the tool input in `info`
    output = env.step(output)
    info = output["metadata"]

    sql_query = info["tool_input"][1]  # SQL query is the second element in tool_input

    # assert it matches the parsed ground truth
    assert sql_query == parse_ground_truth


# Additional test for SQL execution
def test_sql_execution(mock_db_file, mock_sqlite_connection):
    extras = {
        "reward_spec": {
            "method": "rule",
            "ground_truth": "SELECT name FROM employees WHERE id = 1;",
        },
        "max_turns": 2,
        "db_id": "test_db",
        "data": "spider",
    }
    env = skyrl_gym.make(
        "text2sql",
        env_config=DictConfig({"db_path": "/home/ray/default/sql_data"}),
        extras=extras,
    )
    # Skip init() since it's not used in this test

    # Test a valid SQL query
    output = "<sql>SELECT name FROM employees WHERE id = 1;</sql>"
    output = env.step(output)
    observation = output["observations"]

    # Verify the SQL was executed
    mock_sqlite_connection.assert_called_once()

    # Verify the observation contains the expected result
    assert "John Doe" in observation[0]["content"]


@pytest.mark.parametrize(
    "output, expected_valid, description",
    [
        # Valid case
        (
            "<think>I need to query the database</think>\n<solution>SELECT * FROM users;</solution>",
            True,
            "valid_format",
        ),
        # Missing think tag (should fail)
        ("<solution>SELECT * FROM users;</solution>", False, "missing_think_tag"),
        # Multiple solution start tags
        (
            "<think>thinking</think>\n<solution>SELECT * FROM users;<solution>more text</solution>",
            False,
            "multiple_solution_start_tags",
        ),
        # Multiple solution end tags
        (
            "<think>thinking</think>\n<solution>SELECT * FROM users;</solution>extra</solution>",
            False,
            "multiple_solution_end_tags",
        ),
        # Missing solution start tag
        (
            "<think>thinking</think>\nSELECT * FROM users;</solution>",
            False,
            "missing_solution_start_tag",
        ),
        # Missing solution end tag
        (
            "<think>thinking</think>\n<solution>SELECT * FROM users;",
            False,
            "missing_solution_end_tag",
        ),
        # Solution start at very end with no content after (edge case)
        (
            "<think>thinking</think>\nsome text<solution>",
            False,
            "solution_start_at_end",
        ),
        # Solution contains forbidden tags
        (
            "<think>thinking</think>\n<solution>SELECT * FROM users; <sql>forbidden</sql></solution>",
            False,
            "solution_contains_forbidden_sql_tag",
        ),
        # Solution contains forbidden observation tag
        (
            "<think>thinking</think>\n<solution>SELECT * FROM users; <observation>forbidden</observation></solution>",
            False,
            "solution_contains_forbidden_observation_tag",
        ),
        # Empty solution content
        (
            "<think>thinking</think>\n<solution></solution>",
            True,
            "empty_solution_content",
        ),
        # Multiple think tags (should still work)
        (
            "<think>first thought</think>\n<think>second thought</think>\n<solution>SELECT * FROM users;</solution>",
            True,
            "multiple_think_tags",
        ),
        # Observation followed by think (valid pattern)
        (
            "<observation>some observation</observation>\n<think>thinking</think>\n<solution>SELECT * FROM users;</solution>",
            True,
            "observation_then_think",
        ),
        # Observation NOT followed by think (should fail)
        (
            "<observation>some observation</observation>\nrandom text\n<think>thinking</think>\n<solution>SELECT * FROM users;</solution>",
            False,
            "observation_not_followed_by_think",
        ),
        # Edge cases that previously caused unpacking errors
        # Solution tag at very end
        ("some text<solution>", False, "solution_tag_at_end_no_content"),
        # End tag before start tag
        ("</solution><solution>content</solution>", False, "end_tag_before_start_tag"),
        # Malformed nested tags
        ("<solution><solution>content</solution>", False, "malformed_nested_tags"),
        # Only end tag
        ("some text</solution>", False, "only_end_tag"),
        # Empty string
        ("", False, "empty_string"),
        # Only start tag
        ("<solution>content without end", False, "only_start_tag"),
    ],
)
def test_verify_format_and_extract(output, expected_valid, description):
    """Test verify_format_and_extract function with various edge cases, including those that previously caused unpacking errors."""
    # Should not raise an exception regardless of input
    try:
        is_valid, thoughts, solution_text, _ = verify_format_and_extract(output)

        assert is_valid == expected_valid, f"Failed for case: {description}"

        if expected_valid:
            # If valid, should have thoughts and solution_text
            assert (
                thoughts is not None
            ), f"Expected thoughts for valid case: {description}"
            assert (
                solution_text is not None
            ), f"Expected solution_text for valid case: {description}"
            assert (
                len(thoughts) > 0
            ), f"Expected non-empty thoughts for valid case: {description}"
        else:
            # If invalid, thoughts and solution_text should be None
            assert (
                thoughts is None
            ), f"Expected None thoughts for invalid case: {description}"
            assert (
                solution_text is None
            ), f"Expected None solution_text for invalid case: {description}"
    except Exception as e:
        pytest.fail(
            f"verify_format_and_extract should not raise exception for case {description} with input {repr(output)}, but got: {e}"
        )
