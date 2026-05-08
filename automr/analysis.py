
import pandas as pd
import matplotlib.pyplot as plt


class Analyzer:

    def to_dataframe(self, results):
        df = pd.DataFrame(results)
        df["status"] = df["passed"].apply(lambda x: "PASS" if x else "FAIL")
        return df

    def summary(self, df):
        total = len(df)
        passed = df["passed"].sum()
        return {
            "total_tests": total,
            "passed": int(passed),
            "failed": int(total - passed),
            "pass_rate": float(passed / total * 100)
        }

    def plot_results(self, results, title="MR Analysis"):

        x = [r["param"] for r in results]
        y = [r["transformed"] for r in results]
        c = ["green" if r["passed"] else "red" for r in results]

        plt.figure(figsize=(8,5))
        plt.scatter(x, y, c=c)
        plt.xlabel("Parameter")
        plt.ylabel("Model Output")
        plt.title(title)
        plt.grid()
        plt.show()

    def highlight_failures(self, results):

        fail_points = [r for r in results if not r["passed"]]

        if not fail_points:
            print("No failure regions detected")
            return

        print("Failure points:")
        for f in fail_points:
            print(f"param={f['param']:.2f}, change={f['percent_change']:.2f}%")
