# upload_to_drive.py
import os, sys, json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload(local_path, sa_json_str, folder_id=None):
    sa_json = json.loads(sa_json_str)
    creds = service_account.Credentials.from_service_account_info(sa_json, scopes=['https://www.googleapis.com/auth/drive.file'])
    service = build('drive', 'v3', credentials=creds)
    metadata = {'name': os.path.basename(local_path)}
    if folder_id:
        metadata['parents'] = [folder_id]
    media = MediaFileUpload(local_path, resumable=True)
    f = service.files().create(body=metadata, media_body=media, fields='id').execute()
    return f.get('id')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_to_drive.py <file> [GDRIVE_FOLDER_ID]")
    else:
        local = sys.argv[1]
        folder = sys.argv[2] if len(sys.argv) > 2 else None
        sa_json = os.environ.get('GCP_SA_KEY')
        if sa_json is None:
            raise RuntimeError("Please set GCP_SA_KEY env var (service account JSON content).")
        print("Uploading...")
        fid = upload(local, sa_json, folder)
        print("Uploaded id:", fid)
