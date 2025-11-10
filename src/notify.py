# notify.py
import os, sys, requests
def slack_notify(text):
    url = os.environ.get('SLACK_WEBHOOK')
    if not url:
        print("SLACK_WEBHOOK not set")
        return
    payload = {"text": text}
    r = requests.post(url, json=payload, timeout=20)
    print("slack status:", r.status_code)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python notify.py <csv_path>")
        sys.exit(0)
    txt = open(sys.argv[1]).read()
    slack_notify("Top stocks\n" + txt[:2000])
