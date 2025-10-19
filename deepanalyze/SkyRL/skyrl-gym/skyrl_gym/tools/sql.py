from func_timeout import func_timeout, FunctionTimedOut
from skyrl_gym.tools.core import tool, ToolGroup
import pandas as pd
import sqlite3
import sys
import os


class SQLCodeExecutorToolGroup(ToolGroup):
    def __init__(self, db_file_path: str):
        self.db_path = db_file_path
        super().__init__(name="SQLCodeExecutorToolGroup")

    @tool
    def sql(self, db_id, sql, turns_left, timeout=5) -> str:
        def _execute_sql(db_file, sql):
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")
                cursor.execute(sql)
                execution_res = frozenset(cursor.fetchall())
                conn.rollback()
                conn.close()
                return execution_res
            except Exception as e:
                conn.rollback()
                conn.close()
                return f"Error executing SQL: {str(e)}, db file: {db_file}"

        def _execute_sql_wrapper(db_file, sql, timeout=5) -> str:
            try:
                res = func_timeout(timeout, _execute_sql, args=(db_file, sql))
                if isinstance(res, frozenset):
                    df = pd.DataFrame(res)
                    res = df.to_string(index=False)
                    # NOTE: observation too long, just truncate
                    if len(res) > 9000:
                        # just truncate
                        truncated_df = df.head(50)
                        res = (
                            "Truncated to 50 lines since returned response too long: "
                            + truncated_df.to_string(index=False)
                        )  # or index=True if you want row numbers
                else:
                    res = str(res)

            except KeyboardInterrupt:
                sys.exit(0)
            except FunctionTimedOut:
                res = f"SQL Timeout:\n{sql}"
            except Exception as e:
                res = str(e)

            return res

        # TODO (erictang000): move this logic up into the text2sql env, since this is more specific logic
        reminder_text = f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        if sql is None:
            obs = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
        else:
            db_file = os.path.join(self.db_path, db_id, db_id + ".sqlite")
            obs = _execute_sql_wrapper(db_file, sql, timeout)

        return f"\n\n<observation>{obs}\n{reminder_text}</observation>\n\n"
