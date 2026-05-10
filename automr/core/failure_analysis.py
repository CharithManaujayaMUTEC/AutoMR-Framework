
def detect_failure_regions(results):

    failures = [r for r in results if not r["passed"]]

    if not failures:
        return []

    regions = []
    current_region = [failures[0]["param"]]

    for i in range(1, len(failures)):
        prev = failures[i-1]["param"]
        curr = failures[i]["param"]

        if abs(curr - prev) < 0.05:  # continuous region
            current_region.append(curr)
        else:
            regions.append((min(current_region), max(current_region)))
            current_region = [curr]

    regions.append((min(current_region), max(current_region)))

    return regions
