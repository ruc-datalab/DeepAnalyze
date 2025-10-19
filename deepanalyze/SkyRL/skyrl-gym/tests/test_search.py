"""
uv run --isolated --extra dev pytest tests/test_search.py
"""

import skyrl_gym
import pytest
from unittest.mock import patch
from omegaconf import DictConfig
from skyrl_gym.envs.search.utils import extract_solution, em_check


# Mock search API responses for testing
MOCK_SEARCH_RESULTS = {
    "Who is the president of France?": {
        "result": [
            [
                {
                    "document": {
                        "contents": "Emmanuel Macron is the current President of France, serving since 2017."
                    }
                },
                {
                    "document": {
                        "contents": "France is a republic with a president as head of state."
                    }
                },
            ]
        ]
    },
    "What is the capital of Italy?": {
        "result": [
            [
                {
                    "document": {
                        "contents": "Rome is the capital and largest city of Italy."
                    }
                }
            ]
        ]
    },
    "Invalid query": {"result": []},
}


@pytest.fixture
def mock_search_api():
    """Mock the search API requests."""
    with patch("skyrl_gym.tools.search.call_search_api") as mock_call:

        def mock_search_response(
            retrieval_service_url,
            query,
            topk=3,
            return_scores=True,
            timeout=30,
            log_requests=True,
            session=None,
        ):
            if query in MOCK_SEARCH_RESULTS:
                return MOCK_SEARCH_RESULTS[query], None
            elif query == "timeout_query":
                return None, "Timeout Error: Request timed out"
            elif query == "server_error":
                return None, "Server Error: 500 Internal Server Error"
            else:
                return {"result": []}, None

        mock_call.side_effect = mock_search_response
        yield mock_call


@pytest.fixture
def search_env():
    """Create a search environment for testing."""
    return skyrl_gym.make(
        "search",
        env_config=DictConfig(
            {
                "log_requests": False,
                "search_url": "http://127.0.0.1:8000/retrieve",
                "topk": 3,
                "timeout": 30,
            }
        ),
        extras={
            "reward_spec": {
                "method": "rule",
                "ground_truth": {"target": "Emmanuel Macron"},
            },
            "max_turns": 3,
        },
    )


# =============================================================================
# BASIC SEARCH ENVIRONMENT FUNCTIONALITY TESTS
# =============================================================================


def test_env_initialization(search_env):
    """Test that the environment initializes correctly."""
    assert search_env.ground_truth == {"target": "Emmanuel Macron"}
    assert search_env.max_turns == 3
    assert search_env.turns == 0
    assert search_env.chat_history == []
    assert search_env.tool_group.get_name() == "SearchToolGroup"
    assert search_env.tool_group.get_tool_names() == ["search"]


def test_env_with_custom_config():
    """Test environment with custom configuration."""
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig(
            {
                "log_requests": True,
                "search_url": "http://custom.example.com/search",
                "topk": 5,
                "timeout": 60,
            }
        ),
        extras={
            "reward_spec": {
                "method": "rule",
                "ground_truth": {"target": "Custom Answer"},
            },
            "max_turns": 5,
        },
    )

    assert env.ground_truth == {"target": "Custom Answer"}
    assert env.max_turns == 5
    assert env.tool_group.search_url == "http://custom.example.com/search"
    assert env.tool_group.topk == 5
    assert env.tool_group.timeout == 60


# =============================================================================
# ACTION PARSING FUNCTIONALITY TESTS
# =============================================================================


@pytest.mark.parametrize(
    "action, expected_input",
    [
        # Valid search query
        (
            "<search>Who is the president of France?</search>",
            ["Who is the president of France?"],
        ),
        # Search with extra whitespace
        (
            "<search>  Who is the president of France?  </search>",
            ["  Who is the president of France?  "],
        ),
        # Search with multiline content
        (
            "<search>Who is the president\nof France?</search>",
            ["Who is the president\nof France?"],
        ),
        # No search tags
        ("Just plain text", [None]),
        # Malformed search tag (no closing tag)
        ("<search>Who is the president of France?", [None]),
        # Empty search tag
        ("<search></search>", [""]),
        # Multiple search tags (should match first)
        ("<search>Query 1</search> <search>Query 2</search>", ["Query 1"]),
        # Search with answer tag (should still parse search)
        ("<search>Query</search> <answer>Answer</answer>", ["Query"]),
    ],
)
def test_parse_action(search_env, action, expected_input):
    """Test action parsing with various input formats."""
    query = search_env._parse_action(action)
    assert query == expected_input


# =============================================================================
# ACTION POSTPROCESSING FUNCTIONALITY TESTS
# =============================================================================


@pytest.mark.parametrize(
    "action, expected",
    [
        # Search with extra content after closing tag
        ("<search>Query</search> extra content", "<search>Query</search>"),
        # Answer with extra content after closing tag
        ("<answer>Answer</answer> extra content", "<answer>Answer</answer>"),
        # Both search and answer tags
        (
            "<search>Query</search> <answer>Answer</answer> extra",
            "<search>Query</search>",
        ),
        # Only search tag (no extra content)
        ("<search>Query</search>", "<search>Query</search>"),
        # Only answer tag (no extra content)
        ("<answer>Answer</answer>", "<answer>Answer</answer>"),
        # No special tags
        ("Just plain text", "Just plain text"),
        # end tag before start tag
        ("</search><search>Query</search>", "</search>"),
    ],
)
def test_postprocess_action(search_env, action, expected):
    """Test action postprocessing."""
    result = search_env._postprocess_action(action)
    assert result == expected


# =============================================================================
# EPISODE TERMINATION CONDITIONS TESTS
# =============================================================================


@pytest.mark.parametrize(
    "action, current_turns, max_turns, expected",
    [
        # Answer tag should trigger done
        ("<answer>Emmanuel Macron</answer>", 1, 3, True),
        # Search tag should not trigger done
        ("<search>Who is the president?</search>", 1, 3, False),
        # Max turns reached
        ("<search>Query</search>", 3, 3, True),
        # Max turns not reached, no answer tag
        ("Just text", 2, 3, False),
        # Answer tag with max turns reached
        ("<answer>Answer</answer>", 3, 3, True),
        # Mixed content with answer tag
        ("<search>Query</search> <answer>Answer</answer>", 1, 3, True),
    ],
)
def test_is_done(search_env, action, current_turns, max_turns, expected):
    """Test episode termination conditions."""
    search_env.turns = current_turns
    search_env.max_turns = max_turns

    result = search_env._is_done(action)
    assert result == expected


# =============================================================================
# TOOL EXECUTION WITH MOCKED SEARCH API TESTS
# =============================================================================


def test_successful_search(search_env, mock_search_api):
    """Test successful search execution."""
    action = "<search>Who is the president of France?</search>"

    result = search_env.step(action)

    # Verify API was called correctly
    mock_search_api.assert_called_once()
    call_args = mock_search_api.call_args
    assert call_args[1]["query"] == "Who is the president of France?"
    assert call_args[1]["topk"] == 3

    # Verify result structure
    assert not result["done"]
    assert result["reward"] == 0.0  # No reward for intermediate steps
    assert len(result["observations"]) == 1
    assert result["observations"][0]["role"] == "user"

    # Verify observation contains search results
    observation_content = result["observations"][0]["content"]
    assert "<information>" in observation_content
    assert "</information>" in observation_content
    assert "Emmanuel Macron" in observation_content

    # Verify chat history is updated
    assert len(search_env.chat_history) == 2
    assert search_env.chat_history[0]["role"] == "assistant"
    assert search_env.chat_history[0]["content"] == action
    assert search_env.chat_history[1]["role"] == "user"


def test_search_with_no_results(search_env, mock_search_api):
    """Test search with no results."""
    action = "<search>Invalid query</search>"

    result = search_env.step(action)

    # Verify result structure
    assert not result["done"]
    assert result["reward"] == 0.0
    assert len(result["observations"]) == 1

    # Verify observation contains no results message
    observation_content = result["observations"][0]["content"]
    assert "No search results found" in observation_content


def test_search_timeout_error(search_env, mock_search_api):
    """Test search with timeout error."""
    action = "<search>timeout_query</search>"

    result = search_env.step(action)

    # Verify result structure
    assert not result["done"]
    assert result["reward"] == 0.0
    assert len(result["observations"]) == 1

    # Verify observation contains error message
    observation_content = result["observations"][0]["content"]
    assert "Search error" in observation_content
    assert "Timeout Error" in observation_content


def test_search_server_error(search_env, mock_search_api):
    """Test search with server error."""
    action = "<search>server_error</search>"

    result = search_env.step(action)

    # Verify result structure
    assert not result["done"]
    assert result["reward"] == 0.0
    assert len(result["observations"]) == 1

    # Verify observation contains error message
    observation_content = result["observations"][0]["content"]
    assert "Search error" in observation_content
    assert "Server Error" in observation_content


def test_invalid_search_parsing(search_env, mock_search_api):
    """Test search with invalid parsing (None query)."""
    action = "<search>Invalid query"  # No closing tag

    result = search_env.step(action)

    # Should not call the API with None query
    mock_search_api.assert_not_called()

    # Should return empty result
    observation_content = result["observations"][0]["content"]
    assert "<information></information>" in observation_content


# =============================================================================
# REWARD COMPUTATION AND SCORING TESTS
# =============================================================================


@pytest.mark.parametrize(
    "action, ground_truth, expected_reward, expected_done",
    [
        # Correct answer
        ("<answer>Emmanuel Macron</answer>", {"target": "Emmanuel Macron"}, 1.0, True),
        # Incorrect answer
        ("<answer>Nicolas Sarkozy</answer>", {"target": "Emmanuel Macron"}, 0.0, True),
        # Search action (not done)
        (
            "<search>Who is the president?</search>",
            {"target": "Emmanuel Macron"},
            0.0,
            False,
        ),
        # Answer with extra whitespace
        (
            "<answer>  Emmanuel Macron  </answer>",
            {"target": "Emmanuel Macron"},
            1.0,
            True,
        ),
        # Case insensitive match
        ("<answer>emmanuel macron</answer>", {"target": "Emmanuel Macron"}, 1.0, True),
        # Answer without articles
        (
            "<answer>Emmanuel Macron</answer>",
            {"target": "The Emmanuel Macron"},
            1.0,
            True,
        ),
        # Multiple answer tags (should use first one)
        (
            "<answer>Wrong</answer> <answer>Emmanuel Macron</answer>",
            {"target": "Emmanuel Macron"},
            0.0,
            True,
        ),
        # No answer tag
        ("Just text without answer tag", {"target": "Emmanuel Macron"}, 0.0, False),
    ],
)
def test_reward_computation(action, ground_truth, expected_reward, expected_done):
    """Test reward computation for different scenarios."""
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig(
            {
                "log_requests": False,
                "search_url": "http://127.0.0.1:8000/retrieve",
                "topk": 3,
                "timeout": 30,
            }
        ),
        extras={
            "reward_spec": {"method": "rule", "ground_truth": ground_truth},
            "max_turns": 3,
        },
    )

    result = env.step(action)

    assert result["done"] == expected_done
    assert result["reward"] == expected_reward


# =============================================================================
# COMPLETE EPISODE SCENARIOS TESTS
# =============================================================================


def test_successful_search_and_answer(mock_search_api):
    """Test a complete successful episode."""
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig(
            {
                "log_requests": False,
                "search_url": "http://127.0.0.1:8000/retrieve",
                "topk": 3,
                "timeout": 30,
            }
        ),
        extras={
            "reward_spec": {
                "method": "rule",
                "ground_truth": {"target": "Emmanuel Macron"},
            },
            "max_turns": 3,
        },
    )

    # Step 1: Search
    result1 = env.step("<search>Who is the president of France?</search>")
    assert not result1["done"]
    assert result1["reward"] == 0.0
    assert len(result1["observations"]) == 1
    assert "Emmanuel Macron" in result1["observations"][0]["content"]

    # Step 2: Answer
    result2 = env.step("<answer>Emmanuel Macron</answer>")
    assert result2["done"]
    assert result2["reward"] == 1.0
    assert len(result2["observations"]) == 0  # No observations when done

    # Verify chat history
    assert len(env.chat_history) == 3
    assert (
        env.chat_history[0]["content"]
        == "<search>Who is the president of France?</search>"
    )
    assert env.chat_history[2]["content"] == "<answer>Emmanuel Macron</answer>"


def test_max_turns_reached(mock_search_api):
    """Test episode termination when max turns reached."""
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig(
            {
                "log_requests": False,
                "search_url": "http://127.0.0.1:8000/retrieve",
                "topk": 3,
                "timeout": 30,
            }
        ),
        extras={
            "reward_spec": {
                "method": "rule",
                "ground_truth": {"target": "Emmanuel Macron"},
            },
            "max_turns": 2,
        },
    )

    # Step 1: Search
    result1 = env.step("<search>Who is the president of France?</search>")
    assert not result1["done"]
    assert result1["reward"] == 0.0

    # Step 2: Another search (should terminate due to max turns)
    result2 = env.step("<search>More info about France?</search>")
    assert result2["done"]
    assert result2["reward"] == 0.0  # No correct answer, so no reward


def test_search_then_wrong_answer(mock_search_api):
    """Test episode with search followed by wrong answer."""
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig(
            {
                "log_requests": False,
                "search_url": "http://127.0.0.1:8000/retrieve",
                "topk": 3,
                "timeout": 30,
            }
        ),
        extras={
            "reward_spec": {
                "method": "rule",
                "ground_truth": {"target": "Emmanuel Macron"},
            },
            "max_turns": 3,
        },
    )

    # Step 1: Search
    result1 = env.step("<search>Who is the president of France?</search>")
    assert not result1["done"]
    assert result1["reward"] == 0.0

    # Step 2: Wrong answer
    result2 = env.step("<answer>Nicolas Sarkozy</answer>")
    assert result2["done"]
    assert result2["reward"] == 0.0


# =============================================================================
# UTILITY FUNCTIONS USED BY SEARCH ENVIRONMENT TESTS
# =============================================================================


@pytest.mark.parametrize(
    "solution_str, expected",
    [
        # Basic answer extraction
        ("<answer>Emmanuel Macron</answer>", "Emmanuel Macron"),
        # Answer with whitespace
        ("<answer>  Emmanuel Macron  </answer>", "Emmanuel Macron"),
        # Multiple answer tags (should return last one)
        ("<answer>Wrong</answer> <answer>Emmanuel Macron</answer>", "Emmanuel Macron"),
        # No answer tags
        ("No answer here", None),
        # Empty answer
        ("<answer></answer>", ""),
        # Answer with newlines
        ("<answer>Emmanuel\nMacron</answer>", "Emmanuel\nMacron"),
        # Malformed answer tag
        ("<answer>Emmanuel Macron", None),
    ],
)
def test_extract_solution(solution_str, expected):
    """Test solution extraction from text."""
    result = extract_solution(solution_str)
    assert result == expected


@pytest.mark.parametrize(
    "prediction, golden_answers, expected",
    [
        # Exact match
        ("Emmanuel Macron", "Emmanuel Macron", 1),
        # Case insensitive
        ("emmanuel macron", "Emmanuel Macron", 1),
        # With articles
        ("The Emmanuel Macron", "Emmanuel Macron", 1),
        # With punctuation
        ("Emmanuel Macron!", "Emmanuel Macron", 1),
        # No match
        ("Nicolas Sarkozy", "Emmanuel Macron", 0),
        # List of golden answers
        ("Emmanuel Macron", ["Emmanuel Macron", "Macron"], 1),
        # No match in list
        ("Sarkozy", ["Emmanuel Macron", "Macron"], 0),
    ],
)
def test_em_check(prediction, golden_answers, expected):
    """Test exact match checking."""
    result = em_check(prediction, golden_answers)
    assert result == expected


# =============================================================================
# ERROR HANDLING SCENARIOS TESTS
# =============================================================================


def test_missing_reward_spec():
    """Test that missing reward_spec raises an error."""
    with pytest.raises(AssertionError, match="reward_spec field is required"):
        skyrl_gym.make(
            "search",
            env_config=DictConfig(
                {
                    "log_requests": False,
                    "search_url": "http://127.0.0.1:8000/retrieve",
                    "topk": 3,
                    "timeout": 30,
                }
            ),
            extras={"max_turns": 3},  # Missing reward_spec
        )


def test_missing_ground_truth():
    """Test that missing ground_truth raises an error."""
    with pytest.raises(
        AssertionError, match="ground_truth is required in reward_spec field"
    ):
        skyrl_gym.make(
            "search",
            env_config=DictConfig(
                {
                    "log_requests": False,
                    "search_url": "http://127.0.0.1:8000/retrieve",
                    "topk": 3,
                    "timeout": 30,
                }
            ),
            extras={
                "reward_spec": {"method": "rule"},
                "max_turns": 3,
            },  # Missing ground_truth
        )


def test_tool_execution_exception(search_env):
    """Test tool execution when an exception occurs."""
    # Mock the tool execution to raise an exception
    with patch.object(
        search_env.tool_group,
        "execute_tool",
        side_effect=Exception("Tool execution failed"),
    ):
        result = search_env.step("<search>Test query</search>")

        # Should handle the exception gracefully
        assert not result["done"]
        assert result["reward"] == 0.0
        assert len(result["observations"]) == 1
        assert "Tool execution failed" in result["observations"][0]["content"]
