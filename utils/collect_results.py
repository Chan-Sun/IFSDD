import os
import sys
root_path = os.path.dirname(__file__)+"/.."
sys.path.append(root_path)
import json 
import numpy as np
from collections import defaultdict
from main.split import SPLIT
import numpy as np

def read_json(json_path):
    with open(json_path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

def GetMeanResult(json_path,class_list):
    result_list = []
    with open(json_path, 'r') as log_file:
        for line in log_file:
            info = json.loads(line.strip())
            if "mode" in info.keys():
                if info["mode"] == "val":
                    result_list.append(info)
            else:
                continue

    result_keys = [f"bbox_mAP_{classname}"for classname in class_list]
    result_keys.append("bbox_mAP")
    final_result = {keys:0 for keys in result_keys}
    for results in result_list[-3:]:
        for key in result_keys:
            if key in results.keys():
                final_result[key]+=results[key]
    for key, value in final_result.items():
        final_result[key] = round(value/3,4)
    return final_result
    # return list(final_result.values())

def GetBestResult(json_path,class_list):
    result_list = []
    with open(json_path, 'r') as log_file:
        for line in log_file:
            info = json.loads(line.strip())
            if "mode" in info.keys():
                if info["mode"] == "val":
                    result_list.append(info)
            else:
                continue

    result_keys = [f"bbox_mAP_{classname}"for classname in class_list]
    result_keys.append("bbox_mAP")
    
    final_result = {keys:0 for keys in result_keys}
    
    best_result = 0
    best_results_dict = {}
    
    for results in result_list:
        aver_result = results["bbox_mAP"]
        if aver_result > best_result:
            best_result=aver_result
            best_results_dict = results
            
    for key in result_keys:
        if key in best_results_dict.keys():
            final_result[key]+=best_results_dict[key]
    return final_result

def GetBestMeanResult(json_path,class_list):
    result_list = []
    with open(json_path, 'r') as log_file:
        for line in log_file:
            info = json.loads(line.strip())
            if "mode" in info.keys():
                if info["mode"] == "val":
                    result_list.append(info)
            else:
                continue

    result_keys = [f"bbox_mAP_{classname}"for classname in class_list]
    result_keys.append("bbox_mAP")
    
    final_result = {keys:0 for keys in result_keys}
    
    best_result = 0
    best_idx = 0
    
    for idx,results in enumerate(result_list):
        aver_result = results["bbox_mAP"]
        if aver_result > best_result:
            best_result=aver_result
            best_idx = idx
    
    if best_idx == 0:
        best_idx_list = [0,1,2]
    elif best_idx == 7:
        best_idx_list = [5,6,7]
    else:
        best_idx_list = [best_idx-1,best_idx,best_idx+1]

    for idx in best_idx_list:
        results = result_list[idx]
        for key in result_keys:
            if key in results.keys():
                final_result[key]+=results[key]
    for key, value in final_result.items():
        final_result[key] = round(value/3,4)
    return final_result 

work_dir = "./work_dir/"
dataset = "DeepPCB"

# method_list = ["fsce","fsdetview","mpsr","tfa","dkan"]

method_list = ["dkan"]
split = 3
# shot_list = [5,10,30]
shot_list = [5]

for split in [3]:
# for split in [2,3]:
    
    classname = SPLIT[dataset][f"ALL_CLASSES_SPLIT{split}"]
    base_classname = SPLIT[dataset][f"BASE_CLASSES_SPLIT{split}"]
    novel_classname = SPLIT[dataset][f"NOVEL_CLASSES_SPLIT{split}"]
    cls_string = "|split|shot|method|"

    for cls in classname:
        cls_string+=f"{cls}|"
    cls_string+="aver|"

    print(cls_string)
    for shot in shot_list:    
        for method in method_list:
            fs_folder = f"{work_dir}/{dataset}/{method}/SPLIT{split}_SEED1_{shot}SHOT/ablation"
            result_folder = os.path.join(fs_folder,os.listdir(fs_folder)[1])
            print(result_folder)
            for name in os.listdir(result_folder):
            
                if "log.json" in name:
                    json_path = os.path.join(result_folder,name)
                    classwise_result = GetMeanResult(json_path,classname)
                    # classwise_result = GetBestResult(json_path,classname)
                                        
                    string = f"|split{split}|{shot}shot|{method}|"

                    for result in list(classwise_result.values()):
                        string+=f"{result}|"
                    print(string)
            
# dataset_list = os.listdir(work_dir)
# model_list = os.listdir(os.path.join(work_dir,dataset_list[0]))
# shot_list = [5,10,30]
# for data in dataset_list:
#     classname = SPLIT[data]

#     for shot in shot_list:
#         for model in model_list:
#             dir_name = f"{work_dir}/{data}/{model}/{shot}SHOT/"
#             for name in os.listdir(dir_name):
#                 if "log.json" in name:
#                     json_path = os.path.join(dir_name,name)

#             # json_path = f"{work_dir}/{data}/{model}/{shot}SHOT/eval_results.json"
#             classwise_result = GetMeanResult(json_path,classname)
#             # final_result = GetMeanResult(result_dict,classname)
#             # print(final_result)            
#             # collected_keys = list(result_dict.keys())[7:-7]
#             # classwise_result = [result_dict[key]for key in collected_keys]
#             # mean_result = round(np.mean(classwise_result),4)
#             # classwise_result.append(mean_result)
#             string = f"|{model}|{shot}shot|"
#             for result in classwise_result:
#                 string+=f"{result}|"
#             print(string)