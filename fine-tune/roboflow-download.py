# from roboflow import Roboflow
# rf = Roboflow(api_key="iY69BDryDiCSiQna7rMy")
# project = rf.workspace("partygame4all").project("granite-getaway")
# version = project.version(1)
# dataset = version.download("yolov8")


from roboflow import Roboflow
rf = Roboflow(api_key="iY69BDryDiCSiQna7rMy")
project = rf.workspace("partygame4all").project("granite-getaway")
version = project.version(2)
dataset = version.download("yolov8")
                