
import pandas as pd

class Analyzer:

    def to_dataframe(self, results):
        df = pd.DataFrame(results)
        df["status"] = df["passed"].apply(lambda x: "PASS" if x else "FAIL")
        return df

    def summary(self, df):
        total = len(df)
        passed = df["passed"].sum()

        return {
            "total": total,
            "passed": int(passed),
            "failed": int(total - passed),
            "pass_rate": float(passed / total * 100)
        }

    def save_csv(self, df, path="results.csv"):
        df.to_csv(path, index=False)

    def save_json(self, df, path="results.json"):
        df.to_json(path, orient="records", indent=2)
