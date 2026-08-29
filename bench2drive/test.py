import pickle

yihong_pickle = "submission_yihong.pkl"
ke_pickle = "pad_workspace/exp/ke/ke/06.11_16.47/submission.pkl"

with open(yihong_pickle, "rb") as file:
    yihong_submission = pickle.load(file)

with open(ke_pickle, "rb") as file:
    ke_submission = pickle.load(file)
    
for key in yihong_submission.keys():
    if key == "predictions":
        continue
    print(yihong_submission[key])
    print(ke_submission[key])