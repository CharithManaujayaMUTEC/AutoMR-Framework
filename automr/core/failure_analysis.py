def detect_failure_regions(results, threshold=0.05):

    failures = [r for r in results if not r["passed"]]

    if not failures:
        return []

    #  ensure ordered
    failures = sorted(failures, key=lambda x: x["param"])

    regions = []
    current_region = [failures[0]["param"]]

    for i in range(1, len(failures)):
        prev = failures[i - 1]["param"]
        curr = failures[i]["param"]

        if abs(curr - prev) <= threshold:
            current_region.append(curr)
        else:
            regions.append((min(current_region), max(current_region)))
            current_region = [curr]

    regions.append((min(current_region), max(current_region)))

    return regions