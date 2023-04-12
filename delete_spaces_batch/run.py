from huggingface_hub import HfApi

hf_api = HfApi()

spaces = hf_api.list_spaces(
    author="owkin",
    search="trainer-"
)

for space in spaces:
    hf_api.delete_repo(repo_id=space.id, repo_type="space")
