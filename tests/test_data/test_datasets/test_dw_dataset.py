from unittest.mock import patch

from clasymm.datasets.api_wrappers import DataPlatformReader


@patch('psycopg2.connect')
@patch.object(DataPlatformReader, 'check_object')
@patch.object(DataPlatformReader, 'get_obj_info_by_label_id')
@patch.object(DataPlatformReader, 'get_scene_info_by_scene_id')
def test_dataplatform_dataset(mock_get_scene, mock_get_obj, mock_check_obj,
                              mock_connect):

    mock_get_scene.return_value = 'GEP_20160228_000000_P000_RGB_PS.tif'
    mock_get_obj.return_value = (17291.643, 17887.648, 32, 39, -1.403417, 4292,
                                 13210)
    mock_check_obj.return_value = None
    mock_connect.return_value.cursor.return_value = None

    from clasymm.datasets import DataPlatformDataset

    DataPlatformDataset(
        pipeline=[],
        categories=[
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4292'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4315'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4060'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4066'
            },
            {
                'id': 1,
                'supercategory': '300',
                'name': '373'
            },
            {
                'id': 1,
                'supercategory': '300',
                'name': '364'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4042'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4051'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4042'
            },
            {
                'id': 0,
                'supercategory': '4000',
                'name': '4051'
            },
        ],
        object_list=[str(i + 509402) for i in range(10)],
    )
