import { describe, expect, it } from "vitest";

import type { AvailableAgent } from "@/hooks/useAvailableAgents";
import { isStandaloneHarnessAgent } from "./agentGrouping";

function agent(overrides: Partial<AvailableAgent>): AvailableAgent {
  return {
    id: "ag_test",
    name: "test",
    display_name: "Test",
    description: null,
    harness: null,
    skills: [],
    ...overrides,
  };
}

describe("isStandaloneHarnessAgent", () => {
  it("recognizes catalog-backed harness agents", () => {
    expect(
      isStandaloneHarnessAgent(agent({ name: "droid", display_name: "Droid", harness: "droid" }), {
        droid: "Droid",
      }),
    ).toBe(true);
  });

  it("ignores bundle agents that merely use a labeled harness", () => {
    expect(
      isStandaloneHarnessAgent(
        agent({ name: "polly", display_name: "Polly", harness: "claude-sdk" }),
        { "claude-sdk": "Claude SDK" },
      ),
    ).toBe(false);
  });

  it("requires the harness to come from the server catalog", () => {
    expect(
      isStandaloneHarnessAgent(
        agent({ name: "mystery", display_name: "Mystery", harness: "mystery" }),
        {},
      ),
    ).toBe(false);
  });
});
