from backend.app.services.pipeline.service import PipelineService


def test_hotel_city_mapping_uses_chinese_names() -> None:
    service = PipelineService()
    assert service.build_hotel_inputs("北京市")[0] == "北京"
    assert service.build_hotel_inputs("天津市")[0] == "天津"
    assert service.build_hotel_inputs("杭州市")[0] == "杭州"
    assert service.build_hotel_inputs("上海市")[0] == "上海市"
