from typing import List

from tqdm import tqdm


class DataPlatformReader(object):

    def __init__(self, cursor, categories: List[dict], object_list: List[str]):

        self.cursor = cursor
        self.categories = categories
        self.object_list = object_list
        self.check_object()

        self.dataset = self.make_dataset()

    def execute(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchall()
        return row

    def check_object(self):
        search_query = ('SELECT cx,cy,width,height,angle,label_code_id '
                        'from data_platform.tbl_sia_labels WHERE ')
        condition_query = [f"id='{obj_id}'" for obj_id in self.object_list]
        condition_query = ' OR '.join(condition_query)
        tot_query = search_query + condition_query
        results = self.execute(tot_query)
        assert len(results) == len(self.object_list), (
            'unavailable object is included in object_list')

    def get_obj_info_by_label_id(self, label_id):
        search_query = (
            'SELECT cx,cy,width,height,angle,label_code_id,scene_id '
            'from data_platform.tbl_sia_labels WHERE ')
        condition_query = f"id='{label_id}'"
        tot_query = search_query + condition_query
        return self.execute(tot_query)[0]

    def get_scene_info_by_scene_id(self, scene_id):
        search_query = ('SELECT basepath from data_platform.tbl_sia_scenes '
                        'WHERE ')
        condition_query = f"id='{scene_id}'"
        tot_query = search_query + condition_query
        return self.execute(tot_query)[0][0]

    def get_class_idx(self, name):
        class_idx = next(
            (item['id']
             for item in self.categories if str(item['name']) == str(name)),
            -1,
        )
        assert class_idx != -1, f'{name} does not have an index in categories.'
        return class_idx

    def make_dataset(self):

        data_infos = list()
        for label_id in tqdm(self.object_list):

            info = self.get_obj_info_by_label_id(label_id)
            x, y, w, h, rad, label_code, scene_id = info
            category_id = self.get_class_idx(label_code)
            scene_path = self.get_scene_info_by_scene_id(scene_id)

            data = dict(
                scene_path=scene_path,
                x=x,
                y=y,
                w=w,
                h=h,
                rad=rad,
                category_id=category_id)

            data_infos.append(data)

        return data_infos
