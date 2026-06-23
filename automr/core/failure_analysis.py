import pandas as pd

class FailureAnalyzer:

    def failure_rate_per_mr(self, df):
        """
        % of failures per MR
        """
        summary = df.groupby("mr")["passed"].agg(
            total="count",
            passed="sum"
        ).reset_index()

        summary["failed"] = summary["total"] - summary["passed"]
        summary["failure_rate"] = summary["failed"] / summary["total"]

        return summary.sort_values(by="failure_rate", ascending=False)

    def severity_per_mr(self, df):
        """
        Average severity per MR
        """
        if "severity" not in df.columns:
            df["severity"] = df["difference"].abs()

        return df.groupby("mr")["severity"].mean().sort_values(ascending=False)

    def worst_cases(self, df, top_k=10):
        """
        Top worst violations
        """
        if "severity" not in df.columns:
            df["severity"] = df["difference"].abs()

        return df.sort_values(by="severity", ascending=False).head(top_k)

    def failure_regions(self, df):
        """
        Detect failure regions per MR
        """
        regions = {}

        for mr in df["mr"].unique():
            sub = df[(df["mr"] == mr) & (df["passed"] == False)]

            if sub.empty:
                continue

            params = sorted(sub["param"].values)

            grouped = []
            current = [params[0]]

            for i in range(1, len(params)):
                if abs(params[i] - params[i-1]) < 0.05:
                    current.append(params[i])
                else:
                    grouped.append((min(current), max(current)))
                    current = [params[i]]

            grouped.append((min(current), max(current)))
            regions[mr] = grouped

        return regions
    
    def range_summary(self, df):

        cols = [
            "mr",
            "range_change",
            "range_percent_change",
            "range_passed"
        ]

        return (
            df[cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )