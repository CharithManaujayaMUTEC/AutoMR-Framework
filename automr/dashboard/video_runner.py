from .live_dashboard import LiveDashboard


def run_live_dashboard(
    automr,
    model,
    video_source=0,
    frame_skip=30,
    save_results=True,
    save_violations=True,
    output_dir="results/live_dashboard"
):

    dashboard = LiveDashboard(
        automr=automr,
        model=model,
        frame_skip=frame_skip,
        save_results=save_results,
        save_violations=save_violations,
        output_dir=output_dir
    )

    dashboard.run(video_source)