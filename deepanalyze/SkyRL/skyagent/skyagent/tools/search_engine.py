from skyagent.tools.base import BaseTool, register_tool
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
import os


@register_tool("search_engine")
class SearchEngine(BaseTool):
    name = "search_engine"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }
    google_search_key = os.getenv("GOOGLE_SEARCH_KEY")
    if not google_search_key:
        raise ValueError("GOOGLE_SEARCH_KEY environment variable is required")

    def google_search(self, query: str):
        """
        Performs a Google search using the Serper API.

        Args:
            query (str): The search query string

        Returns:
            str: Formatted search results or error message
        """
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.google_search_key,
            "Content-Type": "application/json",
        }
        data = {
            "q": query,
            "num": 10,
            "extendParams": {
                "country": "en",
                "page": 1,
            },
        }

        for i in range(5):
            try:
                response = requests.post(
                    url, headers=headers, data=json.dumps(data), timeout=10
                )
                results = response.json()
                break
            except Exception as e:
                if i == 4:
                    return f"Google search timeout for query '{query}'. Please try again later."
                continue

        if response.status_code != 200:
            return f"Search API error: {response.status_code} - {response.text}"

        try:
            if "organic" not in results:
                return (
                    f"No results found for query: '{query}'. Use a less specific query."
                )

            web_snippets = []
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace(
                        "Your browser can't play this video.", ""
                    )
                    web_snippets.append(redacted_version)

            content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
            )
            return content
        except Exception as e:
            return f"Error parsing search results for '{query}': {str(e)}"

    def call(self, params: dict, **kwargs) -> Union[str, dict]:
        """
        Executes web search queries.

        Args:
            params (dict): Dictionary containing 'query' (array of strings or single string).
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict: The search results or an error message.
        """
        # Verify required parameters
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        query = params.get("query")
        if not query:
            return {"error": "Query parameter is required."}

        try:
            if isinstance(query, str):
                response = self.google_search(query)
            elif isinstance(query, list):
                with ThreadPoolExecutor(max_workers=3) as executor:
                    response = list(executor.map(self.google_search, query))
                response = "\n=======\n".join(response)
            else:
                return {"error": "Query must be a string or array of strings."}

            return {"results": response}

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}


if __name__ == "__main__":
    # Example usage for testing
    tool = SearchEngine()
    test_params = {"query": ["python programming", "machine learning"]}
    result = tool.call(test_params)
    print("Test Result:", result)
