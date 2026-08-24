from backend.app.services.pipeline.service import PipelineService
from backend.app.services.ranking import build_ranking_service


def test_hotel_city_mapping_uses_chinese_names() -> None:
    service = PipelineService()
    beijing_inputs = service.build_hotel_inputs("北京市")
    assert beijing_inputs[0] == "北京"
    assert len(beijing_inputs) == 5
    assert beijing_inputs[1:3] == ["4000", "20"]
    assert beijing_inputs[4] == "1"
    assert service.build_hotel_inputs("天津市")[0] == "天津"
    assert service.build_hotel_inputs("杭州市")[0] == "杭州"
    assert service.build_hotel_inputs("上海市")[0] == "上海市"


def test_pipeline_ranking_factory_contract_remains_available() -> None:
    assert callable(build_ranking_service().rank_candidates)
