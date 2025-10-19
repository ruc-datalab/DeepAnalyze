from skyagent.tools.base import BaseTool, register_tool
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from skyagent.tools.prompt import EXTRACTOR_PROMPT
import os
from openai import OpenAI
import random


@register_tool("web_browser")
class WebBrowser(BaseTool):
    name = "web_browser"
    description = "Visit webpage(s) and return the summary of the content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["url", "goal"],
    }

    webcontent_maxlength = int(os.getenv("WEBCONTENT_MAXLENGTH", "150000"))
    ignore_jina = os.getenv("IGNORE_JINA", "false").lower() == "true"
    jina_reader_url_prefix = "https://r.jina.ai/"
    jina_api_keys = (
        os.getenv("JINA_API_KEYS", "").split(",") if os.getenv("JINA_API_KEYS") else []
    )

    def call(self, params: dict, **kwargs) -> Union[str, dict]:
        """
        Visits webpage(s) and returns summarized content.

        Args:
            params (dict): Dictionary containing 'url' and 'goal'.
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict: The webpage summary or an error message.
        """
        # Verify required parameters
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        try:
            url = params["url"]
            goal = params["goal"]
        except KeyError as e:
            return {"error": f"Missing required field: {str(e)}"}

        if not goal:
            return {"error": "Goal parameter is required."}

        try:
            if isinstance(url, str):
                response = self.readpage(url, goal)
            elif isinstance(url, list):
                response = []
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(self.readpage, u, goal): u for u in url}
                    for future in as_completed(futures):
                        try:
                            response.append(future.result())
                        except Exception as e:
                            response.append(
                                f"Error fetching {futures[future]}: {str(e)}"
                            )
                response = "\n=======\n".join(response)
            else:
                return {"error": "URL must be a string or array of strings."}

            print(f"Summary Length {len(response)}; Summary Content {response}")
            return {"results": response.strip()}

        except Exception as e:
            return {"error": f"Web browsing failed: {str(e)}"}

    def call_server(self, msgs, max_tries=10):
        """
        Call the OpenAI API server to process webpage content.

        Args:
            msgs: Messages to send to the API
            max_tries: Maximum number of retry attempts

        Returns:
            str: The API response content
        """
        openai_api_key = "EMPTY"
        openai_api_base = os.getenv("WEB_SUMMARY_API_BASE")
        summary_model = os.getenv("WEB_SUMMARY_MODEL")
        assert openai_api_base, "WEB_SUMMARY_API_BASE is not set"
        assert summary_model, "WEB_SUMMARY_MODEL is not set"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=summary_model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.7,
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception as e:
                if attempt == (max_tries - 1):
                    return ""
                continue
        return ""

    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.

        Args:
            url: The URL to read

        Returns:
            str: The webpage content or error message
        """
        if not self.jina_api_keys:
            return "[visit] No Jina API keys available."

        headers = {
            "Authorization": f"Bearer {random.choice(self.jina_api_keys)}",
        }
        max_retries = 3
        timeout = 10

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}", headers=headers, timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        return "[visit] Failed to read page."

    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content and extract relevant information.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: The processed webpage content or error message
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"

            print(service)
            print(content)
            if (
                content
                and not content.startswith("[visit] Failed to read page.")
                and content != "[visit] Empty content."
                and not content.startswith("[document_parser]")
            ):
                content = content[: self.webcontent_maxlength]
                messages = [
                    {
                        "role": "user",
                        "content": EXTRACTOR_PROMPT.format(
                            webpage_content=content, goal=goal
                        ),
                    }
                ]
                raw = self.call_server(messages)

                # Handle long webpage content
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = (
                        int(0.7 * len(content)) if summary_retries > 0 else 25000
                    )
                    status_msg = (
                        (
                            f"[visit] Summary url[{url}] "
                            f"attempt {3 - summary_retries + 1}/3, "
                            f"content length: {len(content)}, "
                            f"truncating to {truncate_length} chars"
                        )
                        if summary_retries > 0
                        else (
                            f"[visit] Summary url[{url}] failed after 3 attempts, "
                            f"final truncation to 25000 chars"
                        )
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT.format(
                        webpage_content=content, goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = self.call_server(messages)
                    summary_retries -= 1

                # Parse JSON response
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        raw = json.loads(raw)
                        break
                    except:
                        raw = self.call_server(messages)
                        parse_retry_times += 1

                # Generate final response
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                        url=url, goal=goal
                    )
                    useful_information += (
                        "Evidence in page: \n"
                        + "The provided webpage content could not be accessed. Please check the URL or file format."
                        + "\n\n"
                    )
                    useful_information += (
                        "Summary: \n"
                        + "The webpage content could not be processed, and therefore, no information is available."
                        + "\n\n"
                    )
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                        url=url, goal=goal
                    )
                    useful_information += (
                        "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                    )
                    useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

                if len(useful_information) < 10 and summary_retries < 0:
                    print(
                        "[visit] Could not generate valid summary after maximum retries"
                    )
                    useful_information = "[visit] Failed to read page"
                return useful_information

            # If we're on the last attempt, return failure message
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(
                    url=url, goal=goal
                )
                useful_information += (
                    "Evidence in page: \n"
                    + "The provided webpage content could not be accessed. Please check the URL or file format."
                    + "\n\n"
                )
                useful_information += (
                    "Summary: \n"
                    + "The webpage content could not be processed, and therefore, no information is available."
                    + "\n\n"
                )
                return useful_information


if __name__ == "__main__":
    # Example usage for testing
    tool = WebBrowser()
    test_params = {
        "url": "https://apple.com",
        "goal": "Find information about the company",
    }
    result = tool.call(test_params)
    print("Test Result:", result)
