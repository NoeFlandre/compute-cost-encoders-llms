Feature: Deterministic project quality gates

  Scenario: Quality gates execute in the mandated order
    Given the project quality runner
    When I inspect the declared QA stages
    Then the stages are ordered as ruff, ty, tests, acceptance tests, architecture checks, CRAP, mutation tests
