import argparse
import json
import os
import random

dataset_root = "/your/data/root"

fewshot_class = {
    "NEU_DET":{
            1: "crazing",2: "inclusion",3: "patches",
            4: "pitted_surface",5: "rolled-in_scale",6: "scratches"},   
    "DeepPCB":{
            1:"open_circuit",2:"short",3:"mouse_bite",
            4:"spur",5:"spurious_copper",6:"pin_hole"}      
        }


def generate_seeds(data_name,data_path,catmap,seeds):
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in catmap.keys()}
    for a in data['annotations']:

        anno[a['category_id']].append(a)

    for i in range(seeds[0], seeds[1]):
        random.seed(i)
        for c in catmap.keys():
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10, 30]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_name, catmap[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f,indent=4, separators=(',', ': '))


def get_save_path_seeds(dataset_name, cls, shots, seed):
    
    fewshot_path = os.path.join(dataset_root,dataset_name,f"annotations/fewshot-split/{seed}")
    os.makedirs(fewshot_path,exist_ok=True)
    prefix = '{}seed_{}shot_{}_trainval'.format(seed,shots, cls)
    return fewshot_path+f"/{prefix}.json"
    
if __name__ == '__main__':
    
    for key, value in fewshot_class.items():

        data_name = key
        data_path = os.path.join(dataset_root,data_name,"annotations/trainval.json")
        catmap = value
        seeds = [1,10]
        
        generate_seeds(data_name,data_path,catmap,seeds)
