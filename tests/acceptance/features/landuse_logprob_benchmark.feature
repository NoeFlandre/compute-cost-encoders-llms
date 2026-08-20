Feature: Binary land-use logprob benchmark

  Scenario: The fixed example produces paired decisions
    Given the approved binary land-use example
    When the benchmark records encoder and LLM log probabilities
    Then both records contain yes and no scores
    And each record contains text-to-logprob timing

  Scenario: A benchmark cannot run without a reserved node
    Given no OAR job allocation is present
    When the benchmark guard is evaluated
    Then execution is refused before model loading
