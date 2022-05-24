from unittest.mock import patch

CATEGORIES = [
    {
        'id': 0,
        'supercategory': 'ship',
        'name': 'ship'
    },
    {
        'id': 1,
        'supercategory': 'car',
        'name': 'large-vehicle'
    },
    {
        'id': 1,
        'supercategory': 'car',
        'name': 'small-vehicle'
    },
    {
        'id': 2,
        'supercategory': 'harbor',
        'name': 'harbor'
    },
]


class ObjectId(object):
    def __init__(self, index):
        self.id = index


class MockScene:
    def __init__(self):
        self.db = [
            dict(scene_path='sample.tif',
                 scene_name='P0000',
                 x_gsd=-1,
                 y_gsd=-1),
            dict(scene_path='sample.tif',
                 scene_name='P0001',
                 x_gsd=-1,
                 y_gsd=-1)
        ]

    def find(self, empty={}):
        return self.db

    def find_one(self, query):
        scene_name = query['scene_name']
        for db in self.db:
            if scene_name == db['scene_name']:
                return db


class MockJson:
    def find(self, empty={}):

        return [{
            '_id': ObjectId('626a59d2499ba060ff4c00a7'),
            'x': 2245,
            'y': 1802,
            'w': 20,
            'h': 10,
            'rad': 1.9513026580749109,
            'class_name': 'small-vehicle'
        }, {
            '_id': ObjectId('626a59d2499ba060ff4c00a8'),
            'x': 1467,
            'y': 2138,
            'w': 12,
            'h': 8,
            'rad': 1.73594500468219,
            'class_name': 'small-vehicle'
        }, {
            '_id': ObjectId('626a59d2499ba060ff4c00a9'),
            'x': 1131,
            'y': 1613,
            'w': 18,
            'h': 7,
            'rad': 0.4124104685069456,
            'class_name': 'small-vehicle'
        }, {
            '_id': ObjectId('626a59d2499ba060ff4c00aa'),
            'x': 155,
            'y': 1865,
            'w': 18,
            'h': 7,
            'rad': 0.21866896131309815,
            'class_name': 'small-vehicle'
        }, {
            '_id': ObjectId('626a59d2499ba060ff4c00ab'),
            'x': 221,
            'y': 1819,
            'w': 20,
            'h': 8,
            'rad': 0.0,
            'class_name': 'small-vehicle'
        }]


@patch('osgeo.gdal.Open')
@patch('pymongo.MongoClient')
def test_mongo_dataset(mock_mongo, mock_gdal):

    from clasymm.datasets import MongoDataset

    mock_mongo.return_value = dict(
        dota2_0=dict(scene_lists=MockScene(), P0000=MockJson()))

    MongoDataset(
        pipeline=[],
        dbname='dota2_0',
        scene_list=['P0000', 'P0000'],
        categories=CATEGORIES,
    )
