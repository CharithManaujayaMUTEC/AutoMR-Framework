"""
Camera source utilities.

Provides a single entry point for opening different video sources.
Supports:

- USB webcams (0, 1, ...)
- Video files
- IP Webcam HTTP streams
- RTSP streams

Returns a standard cv2.VideoCapture object so the rest of the
dashboard code remains unchanged.
"""

import cv2


class CameraSource:

    @staticmethod
    def open(source=0):
        """
        Open a camera or video source.

        Parameters
        ----------
        source : int | str
            Examples:
                0
                1
                "video.mp4"
                "http://192.168.1.10:8080/video"
                "rtsp://..."

        Returns
        -------
        cv2.VideoCapture
        """

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video source: {source}"
            )

        return cap

    @staticmethod
    def ip_webcam(ip, port=8080):
        """
        Return the URL for an Android IP Webcam stream.
        """

        return f"http://{ip}:{port}/video"

    @staticmethod
    def rtsp(ip, path=""):
        """
        Return an RTSP URL.
        """

        if path:
            return f"rtsp://{ip}/{path}"

        return f"rtsp://{ip}"