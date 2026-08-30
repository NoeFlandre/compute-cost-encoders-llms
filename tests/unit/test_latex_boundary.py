import compute_cost_encoders_llms.benchmark.latex as latex_module
import compute_cost_encoders_llms.benchmark.reporting as reporting_module


def test_reporting_reexports_latex_renderers() -> None:
    assert reporting_module.render_latex_document is latex_module.render_latex_document
    assert reporting_module.render_latex_summary is latex_module.render_latex_summary
