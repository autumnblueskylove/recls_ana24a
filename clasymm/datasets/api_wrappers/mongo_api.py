try:
    from pymongo import MongoClient
    IMPORT_MONGO = True
except ImportError:
    IMPORT_MONGO = False
    MongoClient = object
from typing import List, Union

from tqdm import tqdm


class MongoGeoCOCO(object):
    def __init__(self, client: MongoClient, dbname: str,
                 categories: List[dict], scene_list: Union[List[str], int]):

        self.client = client
        self.dbname = dbname
        self.categories = categories
        self.scene_list = scene_list
        self.check_scenes()

        self.dataset = self.make_dataset()

    def check_scenes(self):

        db = self.client[self.dbname]
        collection = db['scene_lists']

        if self.scene_list == -1:
            cursor = collection.find({})
            self.scene_list = [f['scene_name'] for f in cursor]

        cursor = collection.find({})
        scene_names = [f['scene_name'] for f in cursor]
        for scenes in self.scene_list:
            assert scenes in scene_names, (
                f'{scenes} is not indexed in {self.dbname}')

    def get_class_idx(self, name):
        class_idx = next(
            (item['id'] for item in self.categories if item['name'] == name),
            -1,
        )
        assert class_idx != -1
        return class_idx

    def make_dataset(self):

        db = self.client[self.dbname]
        scene_collection = db['scene_lists']
        data_infos = list()

        for scene in tqdm(self.scene_list):

            scene_info = [
                cursor
                for cursor in scene_collection.find({'scene_name': scene})
            ][0]
            scene_path = scene_info['scene_path']

            obj_collection = db[scene]
            search_query = [{
                'class_name': cat['name']
            } for cat in self.categories]
            objs = [
                cursor for cursor in obj_collection.find({'$or': search_query})
            ]
            for obj in objs:
                obj['scene_path'] = scene_path
                obj['category_id'] = self.get_class_idx(obj['class_name'])

            data_infos.extend(objs)

        return data_infos
