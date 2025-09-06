from roboflow import Roboflow

rf = Roboflow(api_key="jEfYRaQcHGaKHlxNvu1d")

# 2) fill these from the dataset URL you’re viewing:
#    URL pattern: https://universe.roboflow.com/<workspace_slug>/<project_slug>
#    Example from your screenshot looked like: /test-dataset/basketball-bs0zc
project = rf.workspace("ai-sjglq").project("basketball-bs0zc-lj816")

# 3) choose a version (usually 1 if there’s only one)
dataset = project.version(1).download("coco")   # <- COCO format

# The SDK returns a local folder path like:
print("Saved to:", dataset.location)

