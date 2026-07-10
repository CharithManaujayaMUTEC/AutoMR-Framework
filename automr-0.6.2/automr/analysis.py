import pandas as pd
class Analyzer:

    def to_dataframe(self, results):
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["status"] = df["passed"].apply(lambda x: "PASS" if x else "FAIL")
        return df

    def summary(self, df):

        if df.empty:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0
            }

        total = len(df)
        passed = df["passed"].sum()
        failed = total - passed

        return {
            "total": total,
            "passed": int(passed),
            "failed": int(failed),
            "pass_rate": float(passed / total * 100)
        }
    
    def prediction_trace(self, df):

        cols = [
            "sample_id",
            "mr",
            "param",
            "original",
            "transformed",
            "difference",
            "percent_change",
            "passed"
        ]

        return df[cols].copy()