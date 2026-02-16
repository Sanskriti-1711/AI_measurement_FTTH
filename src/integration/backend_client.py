import requests


class BackendApiClient:
    """Small client for result callbacks to the backend API."""

    def __init__(self, base_url: str, timeout_sec: int = 30, bearer_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if bearer_token:
            self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})

    def post_job_result(self, job_id: str, status: str, result: dict | None = None, error: dict | None = None):
        url = f"{self.base_url}/api/v1/scans/analyze/{job_id}/result"
        payload: dict = {"status": status}
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error

        response = self.session.post(url, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        if response.content:
            return response.json()
        return {"ok": True}

