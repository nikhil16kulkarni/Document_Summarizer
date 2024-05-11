from huggingface_hub import HfApi, HfFolder

model_name = "document-summarizer2"
username = "nikhil16kulkarni"

# Load your token from the environment or directly from a string
token = HfFolder.get_token()

api = HfApi()

# Create the repo
repo_url = api.create_repo(repo_id=f"{username}/{model_name}", token=token)

# Upload the files
api.upload_folder(
    folder_path="./model",
    repo_id=f"{username}/{model_name}",
    token=token,
    path_in_repo=""
)
